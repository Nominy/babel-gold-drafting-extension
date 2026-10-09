"""Export the pinned ZipEnhancer graph to the browser's static WebGPU schedule.

This is deliberately a model-specific compiler, not an ONNX interpreter. It never
executes neural arithmetic. Attention and Swoosh fusions retain the exact source
DAG and its provenance. The source ONNX and checkpoint provenance are immutable.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import copy
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SOURCE_SHA = "2f18c8f7ff10a2702d6243ce1230db9e73e6804dd6cd7b20d8e191ee06924016"
CHECKPOINT_SHA = "b18896915e27a821585584221d0c0820f35e12145315ae3f1e73ccd5a68d195f"
ROOT = Path(__file__).resolve().parents[1]
CUSTOM_DOMAIN = "babel.zipenhancer"
TYPE_BYTES = {1: 4, 6: 4, 7: 8, 9: 1, 10: 2}
ALIAS_OPS = {"Reshape", "Squeeze", "Unsqueeze"}
CUSTOM_OPS = {"ZipRelativeAttention", "ZipSwoosh"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def attributes(node):
    result = {}
    for attr in node.attribute:
        value = helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        require(isinstance(value, (int, float, str, list, tuple)),
                f"Unsupported attribute {node.name}/{attr.name}")
        result[attr.name] = list(value) if isinstance(value, tuple) else value
    return result


def tensor_types(model):
    result = {}
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        tensor = value.type.tensor_type
        require(tensor.HasField("shape") and all(d.HasField("dim_value") for d in tensor.shape.dim),
                f"Nonstatic tensor: {value.name}")
        record = (tensor.elem_type, tuple(d.dim_value for d in tensor.shape.dim))
        require(record[0] in TYPE_BYTES and all(d >= 0 for d in record[1]),
                f"Unsupported tensor: {value.name}: {record}")
        require(value.name not in result or result[value.name] == record,
                f"Conflicting tensor annotation: {value.name}")
        result[value.name] = record
    for value in model.graph.initializer:
        result[value.name] = (value.data_type, tuple(value.dims))
    for node in model.graph.node:
        for name in [*node.input, *node.output]:
            require(not name or name in result, f"Missing static tensor annotation: {node.name}: {name}")
    return result


def prune_annotations(model):
    active = {x for n in model.graph.node for x in [*n.input, *n.output] if x}
    active.update(v.name for v in [*model.graph.input, *model.graph.output])
    annotations = [v for v in model.graph.value_info if v.name in active]
    del model.graph.value_info[:]
    model.graph.value_info.extend(annotations)


class AttentionRecognizer:
    def __init__(self, model, sources):
        self.model = model
        self.nodes = list(model.graph.node)
        self.order = {n.name: i for i, n in enumerate(self.nodes)}
        self.producer = {out: n for n in self.nodes for out in n.output}
        self.users = defaultdict(list)
        for n in self.nodes:
            for name in n.input:
                self.users[name].append(n)
        self.constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        self.types = tensor_types(model)
        self.sources = sources

    def shape(self, name, expected):
        require(self.types[name] == (TensorProto.FLOAT, tuple(expected)),
                f"Attention shape/type mismatch: {name}, expected {expected}, got {self.types[name]}")

    def node(self, tensor, op, count=None):
        n = self.producer.get(tensor)
        require(n is not None and n.op_type == op and n.domain in {"", "ai.onnx"},
                f"Expected {op} producing {tensor}")
        require(len(n.output) == 1 and (count is None or len(n.input) == count),
                f"Unexpected arity: {n.name}")
        return n

    def literal(self, name):
        require(name in self.constants, f"Expected literal {name}")
        return self.constants[name]

    def sliced(self, tensor, axis, start, stop):
        n = self.node(tensor, "Slice", 5)
        require(attributes(n) == {}, f"Unexpected Slice attributes: {n.name}")
        values = [self.literal(x).reshape(-1).tolist() for x in n.input[1:]]
        size = self.types[n.input[0]][1][axis]
        require(values[0] == [start] and values[2] == [axis] and values[3] == [1]
                and values[1][0] >= stop and min(values[1][0], size) == stop and len(values[1]) == 1,
                f"Unexpected Slice interval: {n.name}: {values}")
        expected = list(self.types[n.input[0]][1])
        expected[axis] = stop - start
        self.shape(tensor, expected)
        return n

    def reshaped(self, tensor, expected):
        n = self.node(tensor, "Reshape", 2)
        require(attributes(n) == {"allowzero": 0}, f"Unexpected Reshape semantics: {n.name}")
        target = self.literal(n.input[1]).reshape(-1).tolist()
        source_shape = self.types[n.input[0]][1]
        resolved = [source_shape[i] if d == 0 else d for i, d in enumerate(target)]
        require(resolved.count(-1) <= 1, f"Invalid reshape literal: {n.name}")
        if -1 in resolved:
            index = resolved.index(-1)
            resolved[index] = math.prod(source_shape) // -math.prod(resolved)
        require(resolved == list(expected) and math.prod(source_shape) == math.prod(expected),
                f"Unexpected reshape literal: {n.name}: {target}")
        self.shape(tensor, expected)
        return n

    def chunk(self, reduce_max):
        require(attributes(reduce_max) == {"axes": [-1], "keepdims": 1},
                f"Unexpected attention max: {reduce_max.name}")
        scores = self.node(reduce_max.input[0], "Add", 2)
        content = self.node(scores.input[0], "MatMul", 2)
        q_slice = self.node(content.input[0], "Slice", 5)
        q, k = q_slice.input[0], content.input[1]
        _, qdims = self.types[q]
        require(len(qdims) == 4 and qdims[0] == 4 and qdims[3] == 12,
                f"Unexpected Q layout: {q}")
        _, batch, length, _ = qdims
        self.shape(k, [4, batch, 12, length])
        start_values = self.literal(q_slice.input[1]).reshape(-1).tolist()
        require(len(start_values) == 1, f"Unexpected query start: {q_slice.name}")
        start = start_values[0]
        stop = min(start + 64, length)
        require(start >= 0 and start % 64 == 0 and stop > start, f"Unexpected query chunk: {q_slice.name}")
        rows = stop - start
        self.sliced(content.input[0], 2, start, stop)
        relative_shape = self.reshaped(scores.input[1], [4, batch, rows, length])
        gather = self.node(relative_shape.input[0], "Gather", 2)
        require(attributes(gather) == {"axis": 2}, f"Unexpected relative gather: {gather.name}")
        flattened = self.reshaped(gather.input[0], [4, batch, rows * (2 * length - 1)])
        relative = self.node(flattened.input[0], "MatMul", 2)
        p_slice = self.sliced(relative.input[0], 2, start, stop)
        p, pos = p_slice.input[0], relative.input[1]
        self.shape(p, [4, batch, length, 4])
        self.shape(pos, [4, 1, 4, 2 * length - 1])
        local_rows = np.arange(rows, dtype=np.int64)[:, None]
        keys = np.arange(length, dtype=np.int64)[None, :]
        expected_indices = (local_rows * (2 * length - 1) + length - 1 - start - local_rows + keys).reshape(-1)
        indices = self.literal(gather.input[1])
        require(indices.dtype == np.int64 and np.array_equal(indices, expected_indices),
                f"Relative-position indices changed: {gather.name}")
        self.shape(relative.output[0], [4, batch, rows, 2 * length - 1])
        self.shape(gather.output[0], [4, batch, rows * length])
        self.shape(content.output[0], [4, batch, rows, length])
        self.shape(scores.output[0], [4, batch, rows, length])
        self.shape(reduce_max.output[0], [4, batch, rows, 1])
        sub_users = self.users[reduce_max.output[0]]
        require(len(sub_users) == 1, f"Unexpected max consumer: {reduce_max.name}")
        subtract = self.node(sub_users[0].output[0], "Sub", 2)
        require(list(subtract.input) == [scores.output[0], reduce_max.output[0]],
                f"Unexpected stable softmax subtraction: {subtract.name}")
        exp_users = self.users[subtract.output[0]]
        require(len(exp_users) == 1, f"Unexpected subtraction consumer: {subtract.name}")
        exp = self.node(exp_users[0].output[0], "Exp", 1)
        sums = [n for n in self.users[exp.output[0]] if n.op_type == "ReduceSum"]
        divs = [n for n in self.users[exp.output[0]] if n.op_type == "Div"]
        require(len(sums) == len(divs) == 1, f"Unexpected softmax consumers: {exp.name}")
        total, divide = sums[0], divs[0]
        require(len(total.input) == 2 and self.literal(total.input[1]).reshape(-1).tolist() == [-1]
                and attributes(total) == {"keepdims": 1}
                and list(divide.input) == [exp.output[0], total.output[0]],
                f"Unexpected softmax reduction/division: {divide.name}")
        for n in [subtract, exp, divide]:
            self.shape(n.output[0], [4, batch, rows, length])
        self.shape(total.output[0], [4, batch, rows, 1])
        for n in [scores, content, relative, subtract, exp, divide]:
            require(attributes(n) == {}, f"Unexpected arithmetic attributes: {n.name}")
        closure = [q_slice, p_slice, relative, flattened, gather, relative_shape,
                   content, scores, reduce_max, subtract, exp, total, divide]
        require(len({n.name for n in closure}) == 13, "Overlapping chunk nodes")
        return {"q": q, "k": k, "p": p, "pos": pos, "start": start, "rows": rows,
                "length": length, "batch": batch, "probability": divide.output[0], "nodes": closure}

    def joined(self, matmuls, heads, batch, length, width):
        leaves = {n.output[0] for n in matmuls}
        candidates = set(leaves)
        concats = {}
        changed = True
        while changed:
            changed = False
            for tensor in list(candidates):
                for n in self.users[tensor]:
                    if n.op_type == "Concat" and n.name not in concats and all(x in candidates for x in n.input):
                        require(attributes(n) == {"axis": 2}, f"Unexpected attention concat: {n.name}")
                        concats[n.name] = n
                        candidates.add(n.output[0])
                        changed = True
        internal_inputs = {x for n in concats.values() for x in n.input}
        roots = candidates - internal_inputs
        require(len(roots) == 1, f"Attention chunks are not one concat tree: {sorted(roots)}")
        output = next(iter(roots))
        def flatten(tensor):
            if tensor in leaves:
                return [tensor]
            n = self.node(tensor, "Concat")
            return [leaf for name in n.input for leaf in flatten(name)]
        require(flatten(output) == [n.output[0] for n in matmuls], "Attention concat reordered query chunks")
        self.shape(output, [heads, batch, length, width])
        users = self.users[output]
        require(len(users) == 1 and users[0].op_type == "Transpose"
                and attributes(users[0]) == {"perm": [2, 1, 0, 3]},
                f"Unexpected post-attention layout: {output}")
        return output, list(concats.values())

    def run(self):
        grouped = defaultdict(list)
        for n in self.nodes:
            if n.op_type == "ReduceMax":
                chunk = self.chunk(n)
                grouped[tuple(chunk[x] for x in ["q", "k", "p", "pos"])].append(chunk)
        require(len(grouped) == 8 and sum(map(len, grouped.values())) == 40,
                "Pinned graph requires eight attention groups / forty chunks")
        removed = set()
        replacements = {}
        provenance = {}
        groups = []
        for projections, chunks in grouped.items():
            chunks.sort(key=lambda c: c["start"])
            length, batch = chunks[0]["length"], chunks[0]["batch"]
            require([c["start"] for c in chunks] == list(range(0, length, 64)), "Incomplete query coverage")
            require(all((c["length"], c["batch"]) == (length, batch) for c in chunks), "Inconsistent chunks")
            score_names = {n.name for c in chunks for n in c["nodes"]}
            consumer_chunks = defaultdict(list)
            head_slices = {}
            for c in chunks:
                users = self.users[c["probability"]]
                require(Counter(n.op_type for n in users) == {"Slice": 1, "MatMul": 2},
                        f"Unexpected attention probability consumers: {c['probability']}")
                first = next(n for n in users if n.op_type == "Slice")
                self.sliced(first.output[0], 0, 0, 1)
                first_users = self.users[first.output[0]]
                require(len(first_users) == 1 and first_users[0].op_type == "MatMul",
                        f"Unexpected head-zero consumer: {first.name}")
                for n in [*first_users, *(u for u in users if u.op_type == "MatMul")]:
                    require(len(n.input) == 2 and n.input[0] in {c["probability"], first.output[0]}
                            and attributes(n) == {}, f"Unexpected value application: {n.name}")
                    h, width = (1, 48) if n.input[0] == first.output[0] else (4, 8)
                    self.shape(n.input[1], [h, batch, length, width])
                    self.shape(n.output[0], [h, batch, c["rows"], width])
                    consumer_chunks[(n.input[1], h, width)].append(n)
                    if h == 1:
                        head_slices[n.name] = first
            require(len(consumer_chunks) == 3, "Expected three distinct sequential value consumers")
            ordered_consumers = sorted(consumer_chunks.items(), key=lambda item: self.order[item[1][0].name])
            require([key[1] for key, _ in ordered_consumers] == [1, 4, 4], "Unexpected consumer stage order")
            group = {"id": f"attention-{len(groups)}", "projections": dict(zip(["q", "k", "p", "pos"], projections)),
                     "scoreSources": sorted(score_names, key=self.order.get), "consumers": []}
            group_removed = set(score_names)
            boundary_outputs = set()
            previous_output = None
            for stage, ((value, heads, width), matmuls) in zip(["nonlinear", "self1", "self2"], ordered_consumers):
                require(len(matmuls) == len(chunks), "Consumer query coverage differs")
                output, concats = self.joined(matmuls, heads, batch, length, width)
                consumer_nodes = [*matmuls, *concats, *(head_slices[n.name] for n in matmuls if n.name in head_slices)]
                own_sources = {n.name for n in consumer_nodes}
                require(not (group_removed & own_sources), "Overlapping attention consumers")
                group_removed.update(own_sources)
                terminal = self.producer[output]
                if previous_output is not None:
                    require(self.depends_on(value, previous_output), "Value consumers are no longer sequential")
                previous_output = output
                fused = helper.make_node("ZipRelativeAttention", [*projections, value], [output],
                                         name=terminal.name, domain=CUSTOM_DOMAIN,
                                         heads=heads, queryHeadDim=12, posHeadDim=4, valueHeadDim=width)
                replacements[terminal.name] = fused
                provenance[terminal.name] = sorted(score_names | own_sources, key=self.order.get)
                group["consumers"].append({"node": terminal.name, "stage": stage, "heads": heads,
                                           "output": output, "sources": sorted(own_sources, key=self.order.get)})
                boundary_outputs.add(output)
            # A matched name is insufficient: every eliminated edge must stay in
            # the verified region, except the three exact joined output cutpoints.
            for name in group_removed:
                for output in self.nodes[self.order[name]].output:
                    require(output not in {v.name for v in self.model.graph.output}, "Fusion removes model output")
                    if output not in boundary_outputs:
                        require(all(n.name in group_removed for n in self.users[output]),
                                f"Unexpected consumer crossing attention closure: {output}")
            require(not (removed & group_removed), "Attention groups overlap")
            removed.update(group_removed)
            groups.append(group)
        require(len(removed) == 704, f"Expected 704 fused source nodes, got {len(removed)}")
        require(Counter(self.sources[n]["placement"] for n in removed) == {"gpu": 624, "tensor-alias": 80},
                "Attention GPU/alias coverage changed")
        retained = [replacements[n.name] if n.name in replacements else n
                    for n in self.nodes if n.name not in removed or n.name in replacements]
        del self.model.graph.node[:]
        self.model.graph.node.extend(retained)
        self.model.opset_import.append(helper.make_opsetid(CUSTOM_DOMAIN, 1))
        prune_annotations(self.model)
        for n in retained:
            provenance.setdefault(n.name, [n.name])
        return groups, provenance, removed

    def depends_on(self, tensor, ancestor):
        pending, seen = [tensor], set()
        while pending:
            current = pending.pop()
            if current == ancestor:
                return True
            if current in seen:
                continue
            seen.add(current)
            if current in self.producer:
                pending.extend(self.producer[current].input)
        return False


def fuse_swoosh(model, sources, provenance):
    nodes = list(model.graph.node)
    order = {n.name: i for i, n in enumerate(nodes)}
    producers = {name: n for n in nodes for name in n.output}
    users = defaultdict(list)
    for n in nodes:
        for name in n.input:
            users[name].append(n)
    types = tensor_types(model)
    constants = {t.name for t in model.graph.initializer}
    outputs = {v.name for v in model.graph.output}
    arities = {"Add": 2, "Sub": 2, "Mul": 2, "Div": 2, "Exp": 1,
               "Log": 1, "Cast": 1, "Equal": 2, "Where": 3}
    modules = {"feed_forward1", "feed_forward2", "feed_forward3", "conv_module1", "conv_module2"}
    candidates = [n for n in nodes if n.op_type == "MatMul" and len(n.name.split("/")) >= 3
                  and n.name.split("/")[-2] == "out_proj" and n.name.split("/")[-3] in modules]
    require(len(candidates) == 40, "Expected forty Swoosh output projections")
    removed, replacements = set(), {}
    for matmul in candidates:
        require(matmul.domain in {"", "ai.onnx"} and len(matmul.input) == 2
                and len(matmul.output) == 1 and attributes(matmul) == {},
                f"Unexpected Swoosh projection: {matmul.name}")
        prefix = matmul.name.rsplit("/", 1)[0] + "/"
        region, external = {}, set()

        def walk(name):
            producer = producers.get(name)
            if producer is None or not producer.name.startswith(prefix) or producer.op_type not in arities:
                external.add(name)
                return
            if producer.name not in region:
                region[producer.name] = producer
                for value in producer.input:
                    walk(value)

        walk(matmul.input[0])
        ordered = sorted(region.values(), key=lambda n: order[n.name])
        require(Counter(n.op_type for n in ordered) ==
                {"Sub": 3, "Exp": 1, "Add": 1, "Log": 1, "Cast": 1, "Equal": 1, "Where": 1, "Mul": 1},
                f"Unexpected pinned Swoosh source chain: {matmul.name}")
        activations = external - constants
        require(len(activations) == 1, f"Swoosh needs one external activation: {matmul.name}")
        activation = next(iter(activations))
        require(activation in types and types[activation][0] == TensorProto.FLOAT
                and math.prod(types[activation][1]) > 1
                and (activation not in producers or not producers[activation].name.startswith(prefix)),
                f"Incomplete Swoosh activation closure: {matmul.name}")
        dims = types[activation][1]
        # Retain all unique scalar buffers, including infinity, rather than
        # evaluating the source formula or embedding nonfinite shader literals.
        inputs = [activation]
        for n in ordered:
            for name in n.input:
                if name in external and name not in inputs:
                    require(name in constants and types[name] == (TensorProto.FLOAT, ()),
                            f"Non-scalar Swoosh boundary: {n.name}/{name}")
                    inputs.append(name)
        require(set(inputs) == external and len(inputs) <= 7,
                f"Swoosh storage binding limit exceeded: {matmul.name}")
        refs = {name: -(i + 1) for i, name in enumerate(inputs)}
        steps = []
        for n in ordered:
            require(n.domain in {"", "ai.onnx"} and len(n.input) == arities[n.op_type]
                    and len(n.output) == 1 and all(name in refs for name in n.input),
                    f"Invalid Swoosh source DAG: {n.name}")
            require(provenance.get(n.name) == [n.name] and sources[n.name]["placement"] == "gpu"
                    and sources[n.name]["op"] == n.op_type,
                    f"Swoosh source provenance changed: {n.name}")
            attrs = attributes(n)
            input_types = [types[name][0] for name in n.input]
            output_type = TensorProto.FLOAT
            if n.op_type == "Cast":
                require(set(attrs) == {"to"} and attrs["to"] in {TensorProto.FLOAT, TensorProto.BOOL}
                        and input_types[0] in {TensorProto.FLOAT, TensorProto.BOOL},
                        f"Unsupported Swoosh Cast: {n.name}")
                output_type = attrs["to"]
            else:
                require(attrs == {}, f"Unexpected Swoosh attributes: {n.name}")
                if n.op_type == "Where":
                    require(input_types == [TensorProto.BOOL, TensorProto.FLOAT, TensorProto.FLOAT],
                            f"Invalid Swoosh Where types: {n.name}")
                else:
                    require(all(t == TensorProto.FLOAT for t in input_types),
                            f"Swoosh arithmetic must remain FP32: {n.name}")
                    if n.op_type == "Equal":
                        output_type = TensorProto.BOOL
            require(types[n.output[0]] == (output_type, dims),
                    f"Unexpected Swoosh result shape/type: {n.name}")
            require(n.output[0] not in outputs, f"Swoosh closure contains model output: {n.name}")
            if n.output[0] == matmul.input[0]:
                require(len(users[n.output[0]]) == 1 and users[n.output[0]][0].name == matmul.name,
                        f"Unexpected Swoosh output consumer: {n.name}")
            else:
                require(all(user.name in region for user in users[n.output[0]]),
                        f"Consumer crosses Swoosh closure: {n.name}")
            step = {"source": n.name, "op": n.op_type, "inputs": [refs[name] for name in n.input]}
            if n.op_type == "Cast":
                step["to"] = attrs["to"]
            refs[n.output[0]] = len(steps)
            steps.append(step)
        require(types[matmul.input[0]] == (TensorProto.FLOAT, dims),
                f"Swoosh output is not FP32: {matmul.name}")
        terminal = producers[matmul.input[0]]
        require(not (removed & region.keys()), "Overlapping Swoosh source chains")
        removed.update(region)
        # ONNX cannot represent object-array attributes. Keep the exact IR in
        # a string through precision conversion, decoding only for the plan.
        replacements[terminal.name] = helper.make_node(
            "ZipSwoosh", inputs, [matmul.input[0]], name=terminal.name, domain=CUSTOM_DOMAIN,
            steps=json.dumps(steps, separators=(",", ":"), allow_nan=False), output=refs[matmul.input[0]])
        for name in region:
            del provenance[name]
        provenance[terminal.name] = [step["source"] for step in steps]
    require(len(removed) == 400, "Expected four hundred Swoosh GPU sources")
    retained = [replacements[n.name] if n.name in replacements else n
                for n in nodes if n.name not in removed or n.name in replacements]
    del model.graph.node[:]
    model.graph.node.extend(retained)
    prune_annotations(model)
    return removed


def topological_sort(model):
    nodes = list(model.graph.node)
    producers = {}
    for i, n in enumerate(nodes):
        for name in n.output:
            require(name and name not in producers, f"Duplicate/empty tensor producer: {name}")
            producers[name] = i
    roots = {v.name for v in [*model.graph.input, *model.graph.initializer]}
    dependencies, users = [], defaultdict(list)
    for i, n in enumerate(nodes):
        require(all(not x or x in producers or x in roots for x in n.input), f"Dangling edge: {n.name}")
        deps = {producers[x] for x in n.input if x in producers}
        dependencies.append(len(deps))
        for dep in deps:
            users[dep].append(i)
    ready = [i for i, count in enumerate(dependencies) if not count]
    heapq.heapify(ready)
    ordered = []
    while ready:
        i = heapq.heappop(ready)
        ordered.append(nodes[i])
        for user in users[i]:
            dependencies[user] -= 1
            if not dependencies[user]:
                heapq.heappush(ready, user)
    require(len(ordered) == len(nodes), "Cyclic fused/precision graph")
    del model.graph.node[:]
    model.graph.node.extend(ordered)


def mixed_precision(model, provenance):
    from onnxruntime.transformers import float16

    # Keep complete local nonlinear/normalization arithmetic scopes in FP32,
    # not merely the reduction itself (anchoring and variance are sensitive).
    sensitive_ops = {"ReduceMean", "ReduceSum", "ReduceMax", "Exp", "Log", "Sqrt", "Pow", "Atan", "ZipSwoosh"}
    scopes = {n.name.rsplit("/", 1)[0] for n in model.graph.node if n.op_type in sensitive_ops}
    blocked = {n.name for n in model.graph.node
               if n.op_type in sensitive_ops or (n.name.rsplit("/", 1)[0] in scopes
                                                and n.op_type not in {"MatMul", "Conv"})}
    # ORT normally rounds a shared initializer even on its FP32 branch. Split
    # only mixed-use constants so sensitive scopes retain the original bits.
    constants = {t.name: t for t in model.graph.initializer}
    usages = defaultdict(set)
    for n in model.graph.node:
        for name in n.input:
            if name in constants:
                usages[name].add(n.name in blocked)
    for name, uses in usages.items():
        if uses == {False, True} and constants[name].data_type == TensorProto.FLOAT:
            duplicate = copy.deepcopy(constants[name])
            duplicate.name = name + "/zip-fp32"
            require(duplicate.name not in constants, f"Precision initializer collision: {duplicate.name}")
            model.graph.initializer.append(duplicate)
            for n in model.graph.node:
                if n.name in blocked:
                    for i, value in enumerate(n.input):
                        if value == name:
                            n.input[i] = duplicate.name
    # The converter's min/max knobs assert clipping thresholds. Replace only
    # its scalar conversion hook, restoring it even on failure: IEEE round to
    # nearest/even, including subnormals/zero/inf, with no extra clipping.
    original_conversion = float16.convert_np_to_float16
    def ieee_half(value, min_positive_val=5.96e-8, max_finite_val=65504.0):
        return np.asarray(value).astype(np.float16)
    float16.convert_np_to_float16 = ieee_half
    try:
        model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True,
                                               op_block_list=[], node_block_list=sorted(blocked))
    finally:
        float16.convert_np_to_float16 = original_conversion
    # ORT inserts a down/up cast at *each* blocked node. Bypass those bridges
    # on FP32-to-FP32 edges; otherwise the supposed precision island still
    # rounds every intermediate to half. Original source Cast nodes stay real.
    nodes = list(model.graph.node)
    producers = {x: n for n in nodes for x in n.output}
    replacements = {}
    outputs = {v.name for v in model.graph.output}
    eliminated = set()
    boundary_renames = {}
    for n in nodes:
        if n.name in provenance or n.op_type != "Cast" or attributes(n).get("to") != TensorProto.FLOAT:
            continue
        down = producers.get(n.input[0])
        if down is not None and down.name not in provenance and down.op_type == "Cast" \
                and attributes(down).get("to") == TensorProto.FLOAT16:
            if n.output[0] in outputs:
                # The final producer is already FP32. Preserve its exact bits
                # and the public name instead of rounding through half twice.
                boundary_renames[down.input[0]] = n.output[0]
                eliminated.add(n.name)
            else:
                replacements[n.output[0]] = down.input[0]
                eliminated.add(n.name)
    replacements.update(boundary_renames)
    def resolve(name):
        while name in replacements:
            name = replacements[name]
        return name
    for old, new in boundary_renames.items():
        producer = producers.get(old)
        require(producer is not None and producer.name in provenance,
                f"Unexpected precision boundary producer: {old}")
        for i, name in enumerate(producer.output):
            if name == old:
                producer.output[i] = new
    for n in nodes:
        for i, name in enumerate(n.input):
            n.input[i] = resolve(name)
    nodes = [n for n in nodes if n.name not in eliminated]
    keep = {n.name for n in nodes if n.name in provenance}
    needed = set(outputs)
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n.name in keep or any(x in needed for x in n.output):
                before = len(needed)
                keep.add(n.name)
                needed.update(n.input)
                changed |= len(needed) != before
    nodes = [n for n in nodes if n.name in keep]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    prune_annotations(model)
    return model, sorted(blocked)


def fold_swoosh_precision_boundaries(model, provenance):
    nodes = list(model.graph.node)
    producers = {name: n for n in nodes for name in n.output}
    users = defaultdict(list)
    for n in nodes:
        for name in n.input:
            users[name].append(n)
    types = tensor_types(model)
    outputs = {v.name for v in model.graph.output}
    removed = set()

    def synthetic_cast(node, source_type, target_type):
        return (node is not None and node.name not in provenance and node.op_type == "Cast"
                and node.domain in {"", "ai.onnx"} and len(node.input) == len(node.output) == 1
                and attributes(node) == {"to": target_type}
                and types[node.input[0]][0] == source_type and types[node.output[0]][0] == target_type
                and types[node.input[0]][1] == types[node.output[0]][1])

    for n in nodes:
        if n.op_type != "ZipSwoosh":
            continue
        # The shader loads half storage into FP32 once and rounds only when
        # storing its result. Original source Cast steps remain inside the IR.
        up = producers.get(n.input[0])
        if synthetic_cast(up, TensorProto.FLOAT16, TensorProto.FLOAT) \
                and up.output[0] not in outputs and len(users[up.output[0]]) == 1:
            n.input[0] = up.input[0]
            removed.add(up.name)
        consumers = users[n.output[0]]
        if n.output[0] not in outputs and len(consumers) == 1:
            down = consumers[0]
            if synthetic_cast(down, TensorProto.FLOAT, TensorProto.FLOAT16):
                n.output[0] = down.output[0]
                removed.add(down.name)
    retained = [n for n in nodes if n.name not in removed]
    del model.graph.node[:]
    model.graph.node.extend(retained)
    prune_annotations(model)


def normalized_attributes(node, types):
    attrs = attributes(node)
    if node.op_type == "ZipRelativeAttention":
        return attrs
    if node.op_type == "ZipSwoosh":
        require(set(attrs) == {"steps", "output"} and 2 <= len(node.input) <= 7
                and len(node.output) == 1, f"Invalid fused Swoosh: {node.name}")
        output_type, dims = types[node.output[0]]
        require(output_type in {TensorProto.FLOAT, TensorProto.FLOAT16}
                and types[node.input[0]][0] in {TensorProto.FLOAT, TensorProto.FLOAT16}
                and types[node.input[0]][1] == dims
                and all(types[name] == (TensorProto.FLOAT, ()) for name in node.input[1:]),
                f"Swoosh boundary precision changed: {node.name}")
        return {"steps": json.loads(attrs["steps"]), "output": attrs["output"], "outputType": output_type}
    schema = onnx.defs.get_schema(node.op_type, 17, "")
    for name, attr in schema.attributes.items():
        if attr.default_value.type != onnx.AttributeProto.UNDEFINED:
            default = helper.get_attribute_value(attr.default_value)
            if isinstance(default, bytes):
                default = default.decode("utf-8")
            attrs.setdefault(name, list(default) if isinstance(default, tuple) else default)
    if node.op_type == "Conv":
        rank = len(types[node.input[0]][1]) - 2
        attrs.setdefault("kernel_shape", list(types[node.input[1]][1][2:]))
        attrs.setdefault("strides", [1] * rank)
        attrs.setdefault("dilations", [1] * rank)
        attrs.setdefault("pads", [0] * (rank * 2))
    if node.op_type == "Transpose":
        attrs.setdefault("perm", list(reversed(range(len(types[node.input[0]][1])))))
    if node.op_type in {"ReduceMean", "ReduceMax"}:
        attrs.setdefault("axes", list(range(len(types[node.input[0]][1]))))
    return attrs


def normalized_inputs(node):
    inputs = list(node.input)
    if node.op_type in CUSTOM_OPS:
        return inputs
    formal = onnx.defs.get_schema(node.op_type, 17, "").inputs
    # Empty optional names are missing views, not allocated neural data.
    for parameter in formal[len(inputs):]:
        require(parameter.option == onnx.defs.OpSchema.FormalParameterOption.Optional,
                f"Missing required input: {node.name}/{parameter.name}")
        inputs.append("")
    return inputs


def plan_storage(tensors, nodes, inputs, outputs):
    table = {t["name"]: t for t in tensors}
    roots = {}
    def root(name):
        if name not in roots:
            tensor = table[name]
            roots[name] = root(tensor["aliasOf"]) if "aliasOf" in tensor else name
        return roots[name]
    births, deaths = {}, {}
    for n in nodes:
        for name in n["inputs"]:
            if name:
                r = root(name)
                deaths[r] = max(deaths.get(r, -1), n["id"])
        for name in n["outputs"]:
            r = root(name)
            if "aliasOf" not in table[name]:
                births[r] = n["id"]
                deaths[r] = max(deaths.get(r, -1), n["id"])
    pinned = {root(name) for name in [*inputs, *outputs]}
    pinned.update(t["name"] for t in tensors if "initializer" in t)
    intervals = []
    for name, birth in births.items():
        if name not in pinned:
            t = table[name]
            size = math.prod(t["dims"]) * TYPE_BYTES[t["dataType"]]
            intervals.append((name, birth, deaths[name], size))
    # All lifetimes are known before GPU buffers exist. Place largest buffers
    # first, giving every slot its final capacity immediately rather than
    # accumulating undersized historical allocations. Smaller tensors of any
    # dtype may occupy holes between its previous occupants. Prefer the
    # tightest temporal gap, preserving wider gaps for subsequent intervals.
    slots, starts, occupants = [], [], []
    for name, birth, death, size in sorted(intervals, key=lambda item: (-item[3], item[1], item[0])):
        eligible = []
        for slot in slots:
            slot_id = slot["id"]
            index = bisect_left(starts[slot_id], birth)
            previous_end = occupants[slot_id][index - 1][1] if index else -1
            next_start = starts[slot_id][index] if index < len(starts[slot_id]) else len(nodes)
            # Inclusive operation intervals forbid input/output overlap even
            # when the input's last use is this very output's producer.
            if previous_end < birth and death < next_start:
                eligible.append((next_start - previous_end, slot["byteLength"], slot_id, index))
        if eligible:
            _, _, slot_id, index = min(eligible)
            slot = slots[slot_id]
        else:
            slot_id, index = len(slots), 0
            slot = {"id": slot_id, "byteLength": size}
            slots.append(slot)
            starts.append([])
            occupants.append([])
        table[name]["slot"] = slot_id
        starts[slot_id].insert(index, birth)
        occupants[slot_id].insert(index, (birth, death))
    # Aliases deliberately have aliasOf, not a second storage allocation/slot.
    for n in nodes:
        if n["kind"] == "alias":
            continue
        input_slots = {table[root(x)].get("slot") for x in n["inputs"] if x}
        output_slots = [table[root(x)].get("slot") for x in n["outputs"]]
        require(all(s is None or s not in input_slots for s in output_slots),
                f"In-operation storage overlap: {n['name']}")
        concrete = [s for s in output_slots if s is not None]
        require(len(concrete) == len(set(concrete)), f"Output storage overlap: {n['name']}")
    return slots


def export(args):
    source_bytes = args.source.read_bytes()
    require(digest(source_bytes) == SOURCE_SHA, "Exporter only supports pinned 2f18 ZipEnhancer ONNX")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    require(metadata["sha256"] == SOURCE_SHA and metadata["checkpointSha256"] == CHECKPOINT_SHA
            and metadata["placementGraph"]["sha256"] == SOURCE_SHA, "Source/checkpoint provenance mismatch")
    model = onnx.load_model_from_string(source_bytes)
    require([(op.domain, op.version) for op in model.opset_import] == [("", 17)], "Unexpected pinned opset")
    records = metadata["placementGraph"]["nodes"]
    require(len(records) == len(model.graph.node) == 2329, "Incomplete 2329-node source inventory")
    sources = {r["name"]: {"name": r["name"], "op": r["opType"], "placement": r["placement"]} for r in records}
    require(len(sources) == 2329 and Counter(s["placement"] for s in sources.values())
            == {"gpu": 2115, "tensor-alias": 214}, "Source placement inventory changed")
    for node, record in zip(model.graph.node, records):
        require(node.name == record["name"] and node.op_type == record["opType"]
                and list(node.input) == [x["name"] for x in record["inputs"]]
                and list(node.output) == [x["name"] for x in record["outputs"]],
                f"Metadata graph edge mismatch: {node.name}")
        require((node.op_type in ALIAS_OPS) == (record["placement"] == "tensor-alias"),
                f"Unsupported alias placement: {node.name}")
    require(not any(t.external_data for t in model.graph.initializer), "External weights are not pinned")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    groups, provenance, removed = AttentionRecognizer(model, sources).run()
    removed.update(fuse_swoosh(model, sources, provenance))
    blocked = []
    if args.precision == "mixed-float16":
        model, blocked = mixed_precision(model, provenance)
        fold_swoosh_precision_boundaries(model, provenance)
    topological_sort(model)
    # Custom outputs retain their known source shapes, including converted
    # dtype annotations; standard ONNX inference then covers inserted casts.
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    types = tensor_types(model)
    inputs = [v.name for v in model.graph.input]
    outputs = [v.name for v in model.graph.output]
    require(inputs == ["noisy_mag", "noisy_pha"] and outputs == ["amp_g", "pha_g"], "Unexpected model ABI")
    require(all(types[x] == (1, (1, 201, 641)) for x in [*inputs, *outputs]), "FP32 model ABI changed")
    active = {x for n in model.graph.node for x in [*n.input, *n.output] if x} | set(inputs + outputs)
    initializers = {t.name: t for t in model.graph.initializer if t.name in active}
    tensors = [{"name": name, "dataType": types[name][0], "dims": list(types[name][1])} for name in sorted(active)]
    table = {t["name"]: t for t in tensors}
    weights = bytearray()
    for name in sorted(initializers):
        value = numpy_helper.to_array(initializers[name])
        raw = value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
        expected_size = math.prod(table[name]["dims"]) * TYPE_BYTES[table[name]["dataType"]]
        require(len(raw) == expected_size, f"Initializer raw byte length mismatch: {name}")
        weights.extend(b"\0" * ((-len(weights)) % 16))
        table[name]["initializer"] = {"offset": len(weights), "byteLength": len(raw)}
        weights.extend(raw)
    nodes = []
    for node in model.graph.node:
        original = provenance.get(node.name)
        require(original is not None or node.op_type == "Cast", f"Unmapped synthetic operator: {node.name}")
        alias = node.op_type in ALIAS_OPS
        n = {"id": len(nodes), "name": node.name, "kind": "alias" if alias else "compute",
             "op": "Alias" if alias else node.op_type, "inputs": normalized_inputs(node),
             "outputs": list(node.output), "attributes": {} if alias else normalized_attributes(node, types),
             "sources": original if original is not None else []}
        if original is None:
            n["reason"] = "precision-cast"
        if alias:
            require(len(node.output) == 1 and types[node.output[0]][0] == types[node.input[0]][0]
                    and math.prod(types[node.output[0]][1]) == math.prod(types[node.input[0]][1]),
                    f"Non-storage alias: {node.name}")
            table[node.output[0]]["aliasOf"] = node.input[0]
        if node.op_type == "ZipSwoosh":
            require(n["sources"] == [step["source"] for step in n["attributes"]["steps"]]
                    and all(name in initializers for name in node.input[1:]),
                    f"Swoosh source/scalar coverage changed: {node.name}")
        nodes.append(n)
    slots = plan_storage(tensors, nodes, inputs, outputs)
    covered = {name for n in nodes for name in n["sources"]}
    gpu_covered = {name for n in nodes if n["kind"] == "compute" for name in n["sources"]}
    require(covered == set(sources), "Incomplete original source coverage")
    require({s["name"] for s in sources.values() if s["placement"] == "gpu"} <= gpu_covered,
            "An original GPU source was mapped only to an alias")
    node_table = {n["name"]: n for n in nodes}
    for group in groups:
        for consumer in group["consumers"]:
            node = node_table[consumer["node"]]
            require(node["op"] == "ZipRelativeAttention" and node["outputs"] == [consumer["output"]]
                    and set(node["sources"]) == set(group["scoreSources"] + consumer["sources"]),
                    "Fused consumer coverage changed during precision conversion")
    # Every custom kernel reports its complete original region; shared
    # attention scores count once in the source inventory, not per consumer.
    fused_nodes = [n for n in nodes if n["op"] in CUSTOM_OPS]
    fused_counts = Counter(n["op"] for n in fused_nodes)
    fused_sources = {name for n in fused_nodes for name in n["sources"]}
    require(fused_counts == {"ZipRelativeAttention": 24, "ZipSwoosh": 40},
            "Expected 24 attention consumers and 40 Swoosh applications")
    require(fused_sources == removed, "Custom fusion source coverage changed")
    fused_placements = Counter(sources[name]["placement"] for name in fused_sources)
    retained_placements = Counter(sources[name]["placement"] for name in set(sources) - fused_sources)
    pinned_bytes = sum(math.prod(table[x]["dims"]) * TYPE_BYTES[table[x]["dataType"]] for x in inputs + outputs)
    def aligned(size):
        return ((size + 15) // 16) * 16
    physical_bytes = sum(aligned(s["byteLength"]) for s in slots)
    physical_bytes += sum(aligned(math.prod(t["dims"]) * TYPE_BYTES[t["dataType"]])
                          for t in tensors if "aliasOf" not in t and "slot" not in t)
    plan = {"schema": "babel-zip-webgpu-v1", "checkpointSha256": CHECKPOINT_SHA,
            "sourceGraphSha256": SOURCE_SHA, "precision": args.precision,
            "kernelAbi": "zip-relative-attention-v1+swoosh-v1",
            "weights": {"file": "zipenhancer-webgpu.weights.bin", "sha256": digest(weights), "byteLength": len(weights)},
            "inputs": inputs, "outputs": outputs, "tensors": tensors, "nodes": nodes, "slots": slots,
            "sources": list(sources.values()), "attentionGroups": groups,
            "statistics": {"sourceNodes": len(sources), "sourceGpuNodes": sum(s["placement"] == "gpu" for s in sources.values()),
                           "fusedSourceNodes": len(fused_sources),
                           "gpuGroups": sum(n["kind"] == "compute" for n in nodes),
                           "plannedBytes": physical_bytes}}
    plan_bytes = (json.dumps(plan, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    report = {"schema": plan["schema"], "precision": args.precision, "sourceGraphSha256": SOURCE_SHA,
              "checkpointSha256": CHECKPOINT_SHA, "planSha256": digest(plan_bytes), "weights": plan["weights"],
              "statistics": plan["statistics"], "attentionGroups": len(groups), "queryChunks": 40,
              "fusedConsumers": fused_counts["ZipRelativeAttention"], "customFusionCounts": dict(sorted(fused_counts.items())),
              "fusedGpuSources": fused_placements["gpu"], "fusedAliasSources": fused_placements["tensor-alias"],
              "retainedGpuSources": retained_placements["gpu"], "retainedAliasSources": retained_placements["tensor-alias"],
              "precisionCasts": sum(n.get("reason") == "precision-cast" for n in nodes),
              "fp32Nodes": blocked, "initializerCount": len(initializers), "tensorCount": len(tensors),
              "slotCount": len(slots), "activationSlotBytes": sum(s["byteLength"] for s in slots),
              "boundaryBytes": pinned_bytes, "qualityValidated": False}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / plan["weights"]["file"]).write_bytes(weights)
    (args.out_dir / "zipenhancer-webgpu.plan.json").write_bytes(plan_bytes)
    report_bytes = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    (args.out_dir / "zipenhancer-webgpu.report.json").write_text(report_bytes, encoding="utf-8", newline="\n")
    print(report_bytes, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["float32", "mixed-float16"], default="mixed-float16")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "models/zipenhancer.onnx")
    parser.add_argument("--metadata", type=Path, default=ROOT / "src/core/audio-enhancement-model.json")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
