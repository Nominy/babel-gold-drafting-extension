"""Inference-only, equivalent original Zipformer query-chunk lowering.

All keys and original relative positions participate in each row's softmax.
Only query rows are partitioned; learned modules/parameters and model context are
unchanged. Consumer matmuls consume chunks directly, never joining score matrices.
"""
from types import MethodType

QUERY_ROWS = 64
MAX_TIME_FRAMES = 641
MAX_CONCAT_INPUTS = 7  # WebGPU's default 8 storage bindings: 7 reads + 1 write.
MAX_SCORE_BUFFER_BYTES = max(
    4 * 101 * QUERY_ROWS * (2 * MAX_TIME_FRAMES - 1) * 4,
    256 * MAX_TIME_FRAMES * 201 * 4,  # final dense encoder convolution input
)


def anchored_instance_norm(self, x):
    """Two-pass biased variance, anchored before reductions to avoid cancellation.

    Shifting every spatial value by one same-channel anchor leaves the exact
    mean-centered deviations/variance unchanged, including affine and epsilon.
    """
    delta = x - x[:, :, :1, :1]
    centered = delta - delta.mean(dim=(2, 3), keepdim=True)
    variance = (centered * centered).mean(dim=(2, 3), keepdim=True)
    normalized = centered / (variance + self.eps).sqrt()
    if self.affine:
        normalized = normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]
    return normalized


def lower_attention(model, torch):
    from modelscope.models.audio.ans.zipenhancer_layers.zipformer import (
        RelPositionMultiheadAttentionWeights, SelfAttention, NonlinAttention)

    class AttentionChunks:
        def __init__(self, chunks, heads):
            self.chunks = chunks
            self.heads = heads

        def __getitem__(self, head_slice):
            # The original layer selects the first head for nonlinear attention.
            if head_slice.start != 0 or head_slice.stop != 1:
                raise ValueError("Unexpected original attention head selection")
            return AttentionChunks([chunk[head_slice] for chunk in self.chunks], 1)

        def matmul(self, values):
            return torch.cat([torch.matmul(chunk, values) for chunk in self.chunks], dim=2)

    def weights(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
        if self.training or key_padding_mask is not None or attn_mask is not None:
            raise ValueError("Packaged model is noncausal eval without masks")
        x = self.in_proj(x)
        length, batch, _ = x.shape
        heads, qdim, pdim = self.num_heads, self.query_head_dim, self.pos_head_dim
        q = x[..., :heads * qdim].reshape(length, batch, heads, qdim).permute(2, 1, 0, 3)
        k = x[..., heads * qdim:2 * heads * qdim].reshape(length, batch, heads, qdim).permute(2, 1, 3, 0)
        p = x[..., 2 * heads * qdim:].reshape(length, batch, heads, pdim).permute(2, 1, 0, 3)
        positions = self.linear_pos(pos_emb).reshape(-1, 2 * length - 1, heads, pdim).permute(2, 0, 3, 1)
        columns = torch.arange(length, device=x.device)
        chunks = []
        # Every production input is the same four-second shape. Only query rows
        # are partitioned; every row still attends to every original key.
        for start in range(0, int(length), QUERY_ROWS):
            query = q[:, :, start:start + QUERY_ROWS]
            positional = torch.matmul(p[:, :, start:start + QUERY_ROWS], positions)
            # Flatten only the last two axes so Gather can share one compact
            # row-major index vector across every head and batch. For local row
            # r, flattening adds r * (2 * length - 1) to the unchanged original
            # relative-position column. No neural values or arithmetic change.
            query_rows = query.shape[2]
            local_rows = torch.arange(query_rows, device=x.device)
            relative_columns = length - 1 - (start + local_rows[:, None]) + columns[None, :]
            indices = (local_rows[:, None] * (2 * length - 1) + relative_columns).flatten()
            positional = positional.flatten(2).index_select(2, indices).reshape(heads, batch, query_rows, length)
            scores = torch.matmul(query, k) + positional
            # Same max-subtraction, exp, sum and division as original inplace_softmax.
            scores = scores - scores.max(dim=-1, keepdim=True).values
            scores = scores.exp()
            chunks.append(scores / scores.sum(dim=-1, keepdim=True))
        return AttentionChunks(chunks, heads)

    def self_attention(self, x, attention):
        length, batch, _ = x.shape
        heads = attention.heads
        values = self.in_proj(x).reshape(length, batch, heads, -1).permute(2, 1, 0, 3)
        output = attention.matmul(values).permute(2, 1, 0, 3).reshape(length, batch, -1)
        return self.whiten(self.out_proj(output))

    def nonlinear_attention(self, x, attention):
        x = self.in_proj(x)
        length, batch, _ = x.shape
        s, values, y = x.chunk(3, dim=2)
        s = self.tanh(self.balancer(s)).reshape(length, batch, self.hidden_channels)
        values = self.identity1(self.whiten1(values) * s)
        values = values.reshape(length, batch, attention.heads, -1).permute(2, 1, 0, 3)
        output = attention.matmul(values).permute(2, 1, 0, 3).reshape(length, batch, -1)
        output = self.identity3(output * self.identity2(y))
        return self.whiten2(self.out_proj(output))

    for module in model.modules():
        if isinstance(module, RelPositionMultiheadAttentionWeights):
            module.forward = MethodType(weights, module)
        elif isinstance(module, SelfAttention):
            module.forward = MethodType(self_attention, module)
        elif isinstance(module, NonlinAttention):
            module.forward = MethodType(nonlinear_attention, module)
        elif isinstance(module, torch.nn.InstanceNorm2d):
            module.forward = MethodType(anchored_instance_norm, module)
    return model


def lower_standard_graph(graph, onnx):
    """Use exact standard operators within default WebGPU storage binding limits.

    ORT 1.29 JSEP does not register PRelu/ConstantOfShape GPU kernels. A PRelu
    slope is already ONNX-broadcastable and the literal fill is graph constant.
    Ordered Concat trees only copy unchanged values; each dispatch binds at most
    seven input buffers and one output buffer, without raising device limits.
    """
    result = []
    for node in graph.graph.node:
        if node.op_type == "PRelu":
            zero = node.name + "/zero"
            negative = node.name + "/negative"
            positive = node.name + "/positive"
            result.extend([
                onnx.helper.make_node("Constant", [], [zero], name=zero,
                                      value=onnx.helper.make_tensor(zero, onnx.TensorProto.FLOAT, [], [0.])),
                onnx.helper.make_node("Greater", [node.input[0], zero], [positive], name=positive),
                onnx.helper.make_node("Mul", [node.input[0], node.input[1]], [negative], name=negative),
                onnx.helper.make_node("Where", [positive, node.input[0], negative], list(node.output), name=node.name),
            ])
        elif node.op_type == "ConstantOfShape":
            value = next((a.t for a in node.attribute if a.name == "value"), None)
            if value is None:
                value = onnx.helper.make_tensor(node.name + "/fill", onnx.TensorProto.FLOAT, [1], [0.])
            literal = node.name + "/fill"
            result.extend([
                onnx.helper.make_node("Constant", [], [literal], name=literal, value=value),
                onnx.helper.make_node("Expand", [literal, node.input[0]], list(node.output), name=node.name),
            ])
        elif node.op_type == "Concat" and len(node.input) > MAX_CONCAT_INPUTS:
            inputs = list(node.input)
            axis = next(a.i for a in node.attribute if a.name == "axis")
            level = 0
            while len(inputs) > MAX_CONCAT_INPUTS:
                grouped = []
                offset = 0
                while (len(grouped) + len(inputs) - offset > MAX_CONCAT_INPUTS
                       and len(inputs) - offset >= MAX_CONCAT_INPUTS):
                    output = f"{node.name}/bounded-concat/{level}/{len(grouped)}"
                    result.append(onnx.helper.make_node(
                        "Concat", inputs[offset:offset + MAX_CONCAT_INPUTS],
                        [output], name=output, axis=axis))
                    grouped.append(output)
                    offset += MAX_CONCAT_INPUTS
                inputs = grouped + inputs[offset:]
                level += 1
            del node.input[:]
            node.input.extend(inputs)
            result.append(node)
        else:
            result.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(result)
    return graph


def specialize_static_graph(graph, onnx):
    """Resolve fixed-shape metadata and exact constant indexing before runtime.

    Neural FP32 arithmetic is never evaluated or fused here. Constant structural
    operators only copy existing values; scalar shape/index arithmetic follows
    the standard ONNX reference evaluator. No-op views need identical inferred
    shapes, and dead nodes/duplicate literals are removed by graph connectivity.
    """
    import hashlib
    import math
    import numpy as np
    from onnx.reference import ReferenceEvaluator

    contract = [1, 201, MAX_TIME_FRAMES]
    for value in [*graph.graph.input, *graph.graph.output]:
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Non-FP32 model boundary: {value.name}")
        dims = value.type.tensor_type.shape.dim
        if len(dims) != len(contract):
            raise ValueError(f"Unexpected model boundary rank: {value.name}")
        for dim, size in zip(dims, contract):
            dim.dim_value = size

    opsets = {op.domain: op.version for op in graph.opset_import}
    structural = {
        "Identity", "Reshape", "Squeeze", "Unsqueeze", "Transpose", "Expand",
        "Slice", "Gather", "GatherElements", "Concat", "ConstantOfShape",
    }
    metadata_ops = {
        "Add", "Sub", "Mul", "Div", "Mod", "Neg", "Equal", "Less", "Greater",
        "LessOrEqual", "GreaterOrEqual", "Where", "Not", "And", "Or", "Min",
        "Max", "Floor", "Ceil", "Cast", "Range", "ReduceProd", "ReduceSum",
    }
    literals = set()
    scalar_literals = set()
    constants = {value.name: onnx.numpy_helper.to_array(value) for value in graph.graph.initializer}
    metadata = {name for name, value in constants.items() if value.dtype.kind in "iub"}
    nodes = []

    def add_constant(name, value, is_metadata):
        value = np.asarray(value)
        constants[name] = value
        literals.add(name)
        graph.graph.initializer.append(onnx.numpy_helper.from_array(value, name))
        if is_metadata:
            metadata.add(name)

    for node in graph.graph.node:
        if node.op_type == "Constant" and node.domain in {"", "ai.onnx"}:
            values = ReferenceEvaluator(node, opsets=opsets).run(None, {})
            for name, value in zip(node.output, values):
                add_constant(name, value, np.asarray(value).dtype.kind in "iub")
                if np.asarray(value).ndim == 0:
                    scalar_literals.add(name)
        else:
            nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)

    def infer_shapes():
        # Re-infer rather than retaining intermediate annotations made obsolete
        # by constant substitution or alias removal.
        del graph.graph.value_info[:]
        inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True, data_prop=True)
        return {
            value.name: tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
            for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
            if value.type.tensor_type.HasField("shape") and all(
                dim.HasField("dim_value") for dim in value.type.tensor_type.shape.dim)
        } | {value.name: tuple(value.dims) for value in graph.graph.initializer}

    while True:
        shapes = infer_shapes()
        nodes = []
        folded = 0
        for node in graph.graph.node:
            values = None
            is_metadata = False
            inputs = [name for name in node.input if name]
            if node.domain not in {"", "ai.onnx"}:
                raise ValueError(f"Nonstandard operator in static graph: {node.domain}")
            if node.op_type in {"Shape", "Size"} and node.input[0] in shapes:
                shape = shapes[node.input[0]]
                if node.op_type == "Size":
                    values = [np.asarray(math.prod(shape), dtype=np.int64)]
                else:
                    start = next((attr.i for attr in node.attribute if attr.name == "start"), 0)
                    end = next((attr.i for attr in node.attribute if attr.name == "end"), len(shape))
                    values = [np.asarray(shape[start:end], dtype=np.int64)]
                is_metadata = True
            elif all(name in constants for name in inputs):
                # Float learned parameters and positional embeddings may be
                # indexed/copied exactly, but never used for offline arithmetic.
                is_metadata = all(name in metadata or name in scalar_literals for name in inputs)
                if node.op_type in structural or node.op_type in metadata_ops and is_metadata:
                    values = ReferenceEvaluator(node, opsets=opsets).run(
                        None, {name: constants[name] for name in inputs})
            if values is None:
                nodes.append(node)
                continue
            for name, value in zip(node.output, values):
                add_constant(name, value, is_metadata or np.asarray(value).dtype.kind in "iub")
            folded += 1
        del graph.graph.node[:]
        graph.graph.node.extend(nodes)
        if not folded:
            break

    shapes = infer_shapes()
    outputs = {value.name for value in graph.graph.output}
    aliases = {}
    nodes = []

    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name

    for node in graph.graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = resolve(name)
        no_op = node.op_type == "Identity"
        if node.op_type in {"Reshape", "Expand", "Squeeze", "Unsqueeze"}:
            no_op = node.input[0] in shapes and shapes.get(node.output[0]) == shapes[node.input[0]]
        elif node.op_type == "Transpose":
            perm = next((list(attr.ints) for attr in node.attribute if attr.name == "perm"), None)
            rank = len(shapes[node.input[0]]) if node.input[0] in shapes else None
            no_op = rank is not None and (perm or list(reversed(range(rank)))) == list(range(rank))
        elif node.op_type == "Concat":
            no_op = len(node.input) == 1
        if no_op and len(node.output) == 1 and node.output[0] not in outputs:
            aliases[node.output[0]] = node.input[0]
        else:
            nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)

    # Only generated literals are deduplicated. Original learned initializer
    # identities remain intact, including the original FP32 parameter bytes.
    seen = {}
    replacements = {}
    initializers = []
    for value in graph.graph.initializer:
        array = constants[value.name]
        key = (array.dtype.str, array.shape, hashlib.sha256(array.tobytes()).digest())
        previous = seen.get(key)
        if value.name in literals and previous is not None and value.name not in outputs:
            replacements[value.name] = previous
        else:
            seen.setdefault(key, value.name)
            initializers.append(value)
    for node in graph.graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = replacements.get(name, name)

    needed = set(outputs)
    nodes = []
    for node in reversed(graph.graph.node):
        if any(name in needed for name in node.output):
            nodes.append(node)
            needed.update(name for name in node.input if name)
    del graph.graph.node[:]
    graph.graph.node.extend(reversed(nodes))
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(value for value in initializers if value.name in needed)
    del graph.graph.value_info[:]
    graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True, data_prop=True)
    remaining_metadata = [node.name for node in graph.graph.node if node.op_type in {"Shape", "Size", "Range"}]
    if remaining_metadata:
        raise ValueError(f"Unresolved static shape/index nodes: {remaining_metadata}")
    return graph
