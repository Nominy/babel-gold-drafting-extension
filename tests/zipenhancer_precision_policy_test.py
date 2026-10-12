"""Full-FP32 export guard regressions; no mixed precision is shippable."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('zip_exporter', ROOT / 'scripts/export-zipenhancer-webgpu.py')
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class Float32PolicyTests(unittest.TestCase):
    def test_rejects_half_and_other_floating_storage(self):
        for kind in (10, 11, 16):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                EXPORTER.require_float32_plan([{'dataType': kind}], [])

    def test_rejects_half_cast_including_fused_steps(self):
        for node in (
            {'op': 'Cast', 'attributes': {'to': 10}},
            {'op': 'ZipSwoosh', 'attributes': {'steps': [{'op': 'Cast', 'to': 10}]}},
            {'op': 'ZipSwoosh', 'attributes': {'outputType': 10}},
            {'op': 'Fusion', 'attributes': {'steps': [
                {'op': 'Fusion', 'attributes': {'steps': [{'op': 'Cast', 'to': 10}]}}]}}
        ):
            with self.subTest(node=node), self.assertRaises(ValueError):
                EXPORTER.require_float32_plan([{'dataType': 1}], [node])


if __name__ == '__main__':
    unittest.main()
