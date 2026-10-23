"""Build scaled per-expert NVFP4 candidates from original BF16 safetensors; never installs a model."""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType, GGUFValueType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

sys.path.insert(0, str(HERE))
from precision import CASES, require, sha, shards, tensors  # noqa: E402

GGML_TYPE_NVFP4 = 40
NVFP4_BLOCK, NVFP4_BYTES = 64, 36
SCALE_CAP = 240.0
MIN_CORRELATION = 0.95

SOURCES = {
    "nemotron": "nemotron-3.5-super-vl",
    "flash-next": "qwen3.8-flash-next",
    "27b": "qwen3.8-27b",
}


class Safetensors:
    """Read-only BF16 views from a sharded safetensors checkpoint."""

    def __init__(self, folder, index):
        self.folder = folder
        self.weight_map = json.loads(index.read_text())["weight_map"]
        self.headers = {}
        self.maps = {}
        self.lock = threading.Lock()

    def _open(self, shard):
        with self.lock:
            if shard not in self.headers:
                path = self.folder / shard
                with path.open("rb") as stream:
                    size = struct.unpack("<Q", stream.read(8))[0]
                    header = (json.loads(stream.read(size)), 8 + size)
                self.maps[shard] = np.memmap(path, dtype=np.uint8, mode="r")
                self.headers[shard] = header
            return self.headers[shard]

    def tensor(self, name):
        shard = self.weight_map[name]
        header, start = self._open(shard)
        info = header[name]
        require(info["dtype"] == "BF16", f"{name} is {info['dtype']}, expected BF16")
        begin, end = info["data_offsets"]
        raw = self.maps[shard][start + begin:start + end]
        return raw.view(np.uint16).reshape(info["shape"])


def bf16_to_f32(values):
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def expert_source(case, safetensors, name):
    """Return (n_expert, rows, cols, fetch(e) -> uint16 [rows, cols]) for one GGUF FFN tensor; dense tensors have one expert."""
    if case == "27b":
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight", name)
        require(match, f"Not a dense FFN tensor: {name}")
        full = safetensors.tensor(f"model.language_model.layers.{match[1]}.mlp.{match[2]}_proj.weight")
        return 1, full.shape[0], full.shape[1], lambda e: full
    match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
    require(match, f"Not an expert tensor: {name}")
    layer, kind = int(match[1]), match[2]
    if case == "nemotron":
        pattern = f"language_model.backbone.layers.{layer}.mixer.experts.{{}}.{kind}_proj.weight"
        first = safetensors.tensor(pattern.format(0))
        count = sum(1 for key in safetensors.weight_map if re.fullmatch(re.escape(pattern).replace(r"\{\}", r"\d+"), key))
        return count, first.shape[0], first.shape[1], lambda e: safetensors.tensor(pattern.format(e))
    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    if kind == "down":
        full = safetensors.tensor(prefix + "down_proj")
        return full.shape[0], full.shape[1], full.shape[2], lambda e: full[e]
    full = safetensors.tensor(prefix + "gate_up_proj")
    require(full.shape[1] % 2 == 0, f"Odd gate_up rows for {name}")
    half = full.shape[1] // 2
    rows = slice(0, half) if kind == "gate" else slice(half, None)
    return full.shape[0], half, full.shape[2], lambda e: full[e, rows]


class Quantizer:
    def __init__(self, runtime):
        os.add_dll_directory(str(runtime))
        self.lib = ctypes.CDLL(str(runtime / "ggml-base.dll"))
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                                 ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ggml_type_name.restype = ctypes.c_char_p
        self.lib.ggml_type_name.argtypes = [ctypes.c_int]
        require(self.lib.ggml_type_name(GGML_TYPE_NVFP4) == b"nvfp4", "Runtime NVFP4 type id mismatch")

    def nvfp4(self, values):
        values = np.ascontiguousarray(values, dtype=np.float32)
        rows, cols = values.shape
        require(cols % NVFP4_BLOCK == 0, f"Row length {cols} is not a multiple of {NVFP4_BLOCK}")
        out = np.empty(rows * cols // NVFP4_BLOCK * NVFP4_BYTES, dtype=np.uint8)
        written = self.lib.ggml_quantize_chunk(GGML_TYPE_NVFP4, values.ctypes.data, out.ctypes.data, 0, rows, cols, None)
        require(written == out.nbytes, "Unexpected NVFP4 output size")
        return out


def scaled_expert(quantizer, weights):
    amax = float(np.max(np.abs(weights)))
    scale = np.float32(amax / (6.0 * SCALE_CAP)) if amax > 0 else np.float32(1.0)
    return quantizer.nvfp4(weights / scale), scale


def reconstruct(data, scale, shape):
    return dequantize(data, GGMLQuantizationType.NVFP4).reshape(shape) * np.float32(scale)


def relative_rmse(estimate, reference):
    return float(np.sqrt(np.mean((estimate - reference) ** 2)) / max(np.sqrt(np.mean(reference ** 2)), 1e-30))


def dims(tensor):
    """(cols, rows, n_expert) for a 3-D expert tensor or a 2-D dense tensor (one expert)."""
    shape = [int(x) for x in tensor.shape]
    return (shape[0], shape[1], shape[2]) if len(shape) == 3 else (shape[0], shape[1], 1)


def select(case, before):
    key, layers, _, expected, pattern = CASES[case]
    selected = [name for name in before if (m := re.fullmatch(pattern, name)) and int(m[1]) < layers]
    require(len(selected) == expected, f"Expected {expected} selected tensors, got {len(selected)}")
    return selected


def check(case, quantizer, safetensors, before, selected, samples):
    """Shape, mapping and encoder diagnostics on sampled experts before any build."""
    names = [selected[0], selected[len(selected) // 2], selected[-1]]
    rows = []
    for name in names:
        tensor = before[name]
        cols, out_rows, n_expert = dims(tensor)
        count, src_rows, src_cols, fetch = expert_source(case, safetensors, name)
        require((count, src_rows, src_cols) == (n_expert, out_rows, cols),
                f"{name}: BF16 shape {(count, src_rows, src_cols)} != GGUF {(n_expert, out_rows, cols)}")
        per_expert = tensor.n_bytes // n_expert
        for e in sorted({0, n_expert // 2, n_expert - 1})[:samples]:
            weights = bf16_to_f32(fetch(e))
            original = dequantize(np.asarray(tensor.data).reshape(-1)[e * per_expert:(e + 1) * per_expert],
                                  tensor.tensor_type).reshape(out_rows, cols)
            r = float(np.corrcoef(weights.reshape(-1), original.reshape(-1))[0, 1])
            require(r >= MIN_CORRELATION, f"{name} expert {e}: mapping correlation {r:.4f} < {MIN_CORRELATION}")
            data, scale = scaled_expert(quantizer, weights)
            plain = quantizer.nvfp4(weights)
            rows.append({
                "tensor": name, "expert": e, "original_type": tensor.tensor_type.name,
                "correlation_with_original": round(r, 6), "scale": float(scale),
                "relative_rmse": {
                    "original_gguf": round(relative_rmse(original, weights), 6),
                    "scaled_nvfp4": round(relative_rmse(reconstruct(data, scale, weights.shape), weights), 6),
                    "plain_nvfp4": round(relative_rmse(reconstruct(plain, 1.0, weights.shape), weights), 6),
                },
            })
    return rows


def build_tensor(quantizer, case, safetensors, tensor, threads):
    cols, rows, n_expert = dims(tensor)
    count, src_rows, src_cols, fetch = expert_source(case, safetensors, tensor.name)
    require((count, src_rows, src_cols) == (n_expert, rows, cols), f"Shape mismatch: {tensor.name}")
    per_expert = rows * cols // NVFP4_BLOCK * NVFP4_BYTES
    data = np.empty(per_expert * n_expert, dtype=np.uint8)
    scales = np.empty(n_expert, dtype=np.float32)

    def one(e):
        block, scale = scaled_expert(quantizer, bf16_to_f32(fetch(e)))
        require(np.isfinite(scale) and scale > 0, f"Invalid scale for {tensor.name} expert {e}")
        data[e * per_expert:(e + 1) * per_expert] = block
        scales[e] = scale

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(one, range(n_expert)))
    return data, scales


def pad(n, alignment):
    return (alignment - n % alignment) % alignment


def tensor_info(name, shape, kind, offset):
    encoded = name.encode()
    info = struct.pack("<Q", len(encoded)) + encoded + struct.pack("<I", len(shape))
    info += b"".join(struct.pack("<Q", int(x)) for x in shape)
    return info + struct.pack("<IQ", int(kind), offset)


def write_shard(reader, source_path, output, layout, total_tensors, produce):
    """Copy one GGUF shard, replacing selected payloads and inserting scale tensors after their weights."""
    raw = np.memmap(source_path, dtype=np.uint8, mode="r")
    kv_start = 24
    kv_end = max(field.offset + sum(part.nbytes for part in field.parts) for field in reader.fields.values())
    if reader.tensors:
        require(kv_end == reader.tensors[0].field.offset, "Metadata end does not match the first tensor info")
    kv = bytearray(raw[kv_start:kv_end])
    if "split.tensors.count" in reader.fields:
        field = reader.fields["split.tensors.count"]
        value = field.parts[-1]
        position = field.offset + sum(part.nbytes for part in field.parts[:-1]) - kv_start
        kv[position:position + value.nbytes] = np.array([total_tensors], dtype=value.dtype).tobytes()
    alignment = reader.alignment
    infos, offset = [], 0
    for name, shape, kind, size, _ in layout:
        infos.append(tensor_info(name, shape, kind, offset))
        offset += size + pad(size, alignment)
    header = bytes(raw[:8]) + struct.pack("<QQ", len(layout), reader.fields["GGUF.kv_count"].parts[0][0])
    head = header + bytes(kv) + b"".join(infos)
    with output.open("xb") as stream:
        stream.write(head + b"\0" * pad(len(head), alignment))
        for name, _, _, size, source in layout:
            payload = produce(name) if source is None else source
            payload = memoryview(np.ascontiguousarray(payload)).cast("B")
            require(payload.nbytes == size, f"Payload size mismatch: {name}")
            for start in range(0, size, 1 << 28):
                stream.write(payload[start:start + (1 << 28)])
            stream.write(b"\0" * pad(size, alignment))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=SOURCES)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--protocol", default="bf16-nvfp4-protocol.json")
    args = parser.parse_args()
    protocol_path = HERE / args.protocol
    protocol = json.loads(protocol_path.read_text())
    require(protocol["parent_protocol_sha256"] == sha(HERE / "protocol.json"), "Parent protocol changed")
    folder = SOURCES[args.case]
    weights = args.source_root / (folder + "-bf16")
    index = weights / "model.safetensors.index.json"
    safetensors = Safetensors(weights, index if index.exists() else args.source_root / folder / "model.safetensors.index.json")
    key = CASES[args.case][0]
    source_files = shards(Path(json.loads((ROOT / "rtx-spark" / "results" / "model_paths.json").read_text())[key]))
    readers = [GGUFReader(str(path)) for path in source_files]
    before = tensors(readers)
    selected = select(args.case, before)
    quantizer = Quantizer(args.runtime)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    diagnostics = check(args.case, quantizer, safetensors, before, selected, 3)
    record = {
        "case": args.case, "protocol_sha256": sha(protocol_path), "script_sha256": sha(Path(__file__)),
        "runtime_ggml_base_sha256": sha(args.runtime / "ggml-base.dll"), "scale_cap": SCALE_CAP,
        "source": protocol["sources"][args.case], "mapping_check": diagnostics,
        "check_seconds": round(time.monotonic() - started, 1),
    }
    print(json.dumps(diagnostics, indent=1), flush=True)
    if args.check_only:
        (args.out / "check.json").write_text(json.dumps(record, indent=2) + "\n")
        return
    chosen = set(selected)
    total = len(before) + len(selected)
    cache, scale_stats = {}, {}

    def produce(name):
        if name not in cache:
            weight = name if name.endswith(".weight") else name.removesuffix(".scale") + ".weight"
            t0 = time.monotonic()
            data, scales = build_tensor(quantizer, args.case, safetensors, before[weight], args.threads)
            cache[weight], cache[weight.removesuffix(".weight") + ".scale"] = data, scales
            scale_stats[weight] = {"min": float(scales.min()), "max": float(scales.max()),
                                   "seconds": round(time.monotonic() - t0, 1)}
            print(f"{weight}: {scale_stats[weight]}", flush=True)
        return cache.pop(name)

    model_dir = args.out / "model"
    model_dir.mkdir()
    outputs = []
    for index, (reader, path) in enumerate(zip(readers, source_files)):
        layout = []
        for tensor in reader.tensors:
            shape = [int(x) for x in tensor.shape]
            if tensor.name in chosen:
                cols, rows, n_expert = dims(tensor)
                size = cols * rows * n_expert // NVFP4_BLOCK * NVFP4_BYTES
                layout.append((tensor.name, shape, GGMLQuantizationType.NVFP4, size, None))
                layout.append((tensor.name.removesuffix(".weight") + ".scale", [n_expert],
                               GGMLQuantizationType.F32, n_expert * 4, None))
            else:
                layout.append((tensor.name, shape, tensor.tensor_type, tensor.n_bytes, tensor.data))
        suffix = f"-{index + 1:05d}-of-{len(readers):05d}.gguf" if len(readers) > 1 else ".gguf"
        output = model_dir / f"{args.case}-bf16-nvfp4{suffix}"
        write_shard(reader, path, output, layout, total, produce)
        outputs.append(output.name)
        print(f"wrote {output.name}", flush=True)
    require(not cache, "Unwritten payloads remain")
    record.update(candidate_filenames=outputs, scales=scale_stats, build_seconds=round(time.monotonic() - started, 1))
    (args.out / "build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
