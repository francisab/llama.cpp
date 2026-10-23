import importlib.util
import json
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNTIME = Path(r"C:\llama.cpp")
sys.path.insert(0, str(RESULTS.parents[1] / "gguf-py"))
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType  # noqa: E402


def load_module(name, folder, filename):
    spec = importlib.util.spec_from_file_location(name, RESULTS / folder / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("bf16_nvfp4", "workload-tradeoffs-20261021", "bf16_nvfp4.py")
precision = load_module("tradeoff_precision_bf16", "workload-tradeoffs-20261021", "precision.py")


def write_source(folder, rng):
    """Three-shard GGUF like Flash-Next: metadata-only shard 1, expert tensor in shard 2, plain tensor in shard 3."""
    tensors = [
        {},
        {"blk.0.ffn_up_exps.weight": rng.standard_normal((2, 4, 64)).astype(np.float32),
         "blk.0.attn_norm.weight": rng.standard_normal(64).astype(np.float32)},
        {"token_embd.weight": rng.standard_normal((3, 64)).astype(np.float32)},
    ]
    paths = []
    for index, group in enumerate(tensors):
        path = folder / f"src-{index + 1:05d}-of-00003.gguf"
        writer = GGUFWriter(str(path), "test")
        writer.add_uint16("split.no", index)
        writer.add_uint16("split.count", 3)
        writer.add_int32("split.tensors.count", 3)
        writer.add_string("test.note", "kept")
        for name, value in group.items():
            writer.add_tensor(name, value)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        paths.append(path)
    return paths


class WriterTests(unittest.TestCase):
    def test_rewrite_keeps_metadata_and_unselected_payloads(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            sources = write_source(tmp, rng)
            readers = [GGUFReader(str(path)) for path in sources]
            before = precision.tensors(readers)
            selected = ["blk.0.ffn_up_exps.weight"]
            payload = np.arange(2 * 4 * 64 // 64 * 36, dtype=np.uint8)
            scales = np.array([0.5, 2.0], dtype=np.float32)
            produced = {"blk.0.ffn_up_exps.weight": payload, "blk.0.ffn_up_exps.scale": scales}
            model = tmp / "out" / "model"
            model.mkdir(parents=True)
            for index, (reader, path) in enumerate(zip(readers, sources)):
                layout = []
                for tensor in reader.tensors:
                    shape = [int(x) for x in tensor.shape]
                    if tensor.name in selected:
                        layout.append((tensor.name, shape, GGMLQuantizationType.NVFP4, payload.nbytes, None))
                        layout.append(("blk.0.ffn_up_exps.scale", [2], GGMLQuantizationType.F32, 8, None))
                    else:
                        layout.append((tensor.name, shape, tensor.tensor_type, tensor.n_bytes, tensor.data))
                builder.write_shard(reader, path, model / f"cand-{index + 1:05d}-of-00003.gguf", layout, 4,
                                    produced.pop)
            (tmp / "out" / "build.json").write_text(json.dumps(
                {"script_sha256": "s", "protocol_sha256": "p", "mapping_check": []}))
            files, audit = precision.audit_prebuilt(model, readers, before, selected)
            self.assertEqual(audit["scale_tensors"], 1)
            self.assertEqual(audit["unchanged_payloads_verified"], 2)
            after = precision.tensors([GGUFReader(str(path)) for path in files])
            np.testing.assert_array_equal(after["blk.0.ffn_up_exps.scale"].data, scales)
            np.testing.assert_array_equal(after["blk.0.ffn_up_exps.weight"].data.reshape(-1), payload)
            self.assertEqual(GGUFReader(str(files[0])).fields["test.note"].contents(), "kept")

    def test_audit_rejects_a_changed_unselected_payload(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            sources = write_source(tmp, rng)
            readers = [GGUFReader(str(path)) for path in sources]
            before = precision.tensors(readers)
            model = tmp / "out" / "model"
            model.mkdir(parents=True)
            for index, (reader, path) in enumerate(zip(readers, sources)):
                layout = []
                for tensor in reader.tensors:
                    shape = [int(x) for x in tensor.shape]
                    data = tensor.data
                    if tensor.name == "token_embd.weight":
                        data = np.array(tensor.data) + 1
                    if tensor.name == "blk.0.ffn_up_exps.weight":
                        layout.append((tensor.name, shape, GGMLQuantizationType.NVFP4, 288, np.zeros(288, np.uint8)))
                        layout.append(("blk.0.ffn_up_exps.scale", [2], GGMLQuantizationType.F32, 8,
                                       np.ones(2, np.float32)))
                    else:
                        layout.append((tensor.name, shape, tensor.tensor_type, tensor.n_bytes, data))
                builder.write_shard(reader, path, model / f"cand-{index + 1:05d}-of-00003.gguf", layout, 4, None)
            (tmp / "out" / "build.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Unselected payload changed"):
                precision.audit_prebuilt(model, readers, before, ["blk.0.ffn_up_exps.weight"])


class MappingTests(unittest.TestCase):
    def test_flash_next_gate_up_split(self):
        full = np.arange(2 * 8 * 64, dtype=np.uint16).reshape(2, 8, 64)

        class Fake:
            def tensor(self, name):
                self.name = name
                return full

        fake = Fake()
        count, rows, cols, fetch = builder.expert_source("flash-next", fake, "blk.3.ffn_gate_exps.weight")
        self.assertEqual((count, rows, cols), (2, 4, 64))
        self.assertEqual(fake.name, "model.language_model.layers.3.mlp.experts.gate_up_proj")
        np.testing.assert_array_equal(fetch(1), full[1, :4])
        _, _, _, fetch = builder.expert_source("flash-next", fake, "blk.3.ffn_up_exps.weight")
        np.testing.assert_array_equal(fetch(1), full[1, 4:])

    def test_dense_27b_is_one_expert(self):
        full = np.arange(4 * 64, dtype=np.uint16).reshape(4, 64)

        class Fake:
            def tensor(self, name):
                self.name = name
                return full

        fake = Fake()
        count, rows, cols, fetch = builder.expert_source("27b", fake, "blk.7.ffn_down.weight")
        self.assertEqual((count, rows, cols), (1, 4, 64))
        self.assertEqual(fake.name, "model.language_model.layers.7.mlp.down_proj.weight")
        np.testing.assert_array_equal(fetch(0), full)
        tensor = SimpleNamespace(shape=np.array([64, 4], dtype=np.uint64))
        self.assertEqual(builder.dims(tensor), (64, 4, 1))

    def test_non_inferiority_needs_both_metrics(self):
        gate = load_module("bf16_reference_gate", "workload-tradeoffs-20261021", "bf16_reference_gate.py")
        control = {"mean_kl_nats": 0.02, "ppl": 6.0}
        self.assertTrue(gate.non_inferior({"mean_kl_nats": 0.02, "ppl": 6.0}, control))
        self.assertFalse(gate.non_inferior({"mean_kl_nats": 0.019, "ppl": 6.01}, control))
        self.assertFalse(gate.non_inferior({"mean_kl_nats": 0.021, "ppl": 5.9}, control))

    def test_bf16_decode(self):
        values = np.array([1.0, -2.5, 0.0], dtype=np.float32)
        bf16 = (values.view(np.uint32) >> 16).astype(np.uint16)
        np.testing.assert_array_equal(builder.bf16_to_f32(bf16), values)


@unittest.skipUnless((RUNTIME / "ggml-base.dll").exists() and platform.machine() == "ARM64",
                     "needs the installed native ARM64 runtime")
class QuantizerTests(unittest.TestCase):
    def test_scaled_expert_round_trip(self):
        quantizer = builder.Quantizer(RUNTIME)
        weights = (np.random.default_rng(3).standard_normal((8, 128)) * 0.02).astype(np.float32)
        data, scale = builder.scaled_expert(quantizer, weights)
        self.assertAlmostEqual(float(scale) * 6 * 240 / float(np.abs(weights).max()), 1.0, places=6)
        error = builder.relative_rmse(builder.reconstruct(data, scale, weights.shape), weights)
        self.assertLess(error, 0.12)
        block_scales = data.reshape(-1, 36)[:, :4]
        self.assertTrue((block_scales < 0x78).all() and (block_scales.max() == 0x77))


if __name__ == "__main__":
    unittest.main()
