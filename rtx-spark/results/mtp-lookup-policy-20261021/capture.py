"""Capture isolated per-round MTP/lookup costs using the existing benchmark runner."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INSTALLED = Path(r"C:\llama.cpp")
SOURCE = Path(r"C:\llama-builds\src-b11756-30380")
BUILD = Path(r"C:\llama-builds\rel-b11756-30380-30640\cpu\bin")
PORT = 9949


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def main():
    if not __debug__:
        raise RuntimeError("Run without -O; diagnostic assertions must remain enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    private = args.private.resolve(strict=True)
    if (private / "before.json").exists() or (private / "mtp").exists() or (private / "hybrid").exists():
        raise FileExistsError("Refusing to replace a previous capture")
    reference_dir = private.parent / "copy-edit-ngram" / "mtp"
    reference = json.loads((reference_dir / "base-1" / "server_U1_copy-edit.json").read_text())
    prior = json.loads((HERE.parent / "prefix-retention-20261021" / "manifest.json").read_text())
    expected = prior["binary_manifest"]
    assert all(digest(INSTALLED / name) == value for name, value in expected.items())
    assert reference["completed"]
    fixture = REPO / "rtx-spark" / "bench" / "copy_edit_suite.py"
    assert digest(fixture) == "1e527f7dc8636f4bc95f1ce952e51b6943cfc19727d41bd0bc63ead817959be9"
    assert digest(INSTALLED / "llama-tokenize.exe") == "6e36d63fc3f304d470e1546370ed1835c3bf8ca9386fd030ed1bcab3e0eeb0d7"
    assert digest(SOURCE / "common" / "ngram-map.cpp") == "e216b3a607204e75d1c9a2e9c61473b5d3abdc95c794d55c6e62ced7f7f1aed7"
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", PORT)) != 0

    runtime = private / "runtime"
    runtime.mkdir(exist_ok=True)
    assert {path.name for path in runtime.iterdir()} <= set(expected)
    for name in expected:
        shutil.copy2(INSTALLED / name, runtime / name)
    for name in ("llama-common.dll", "llama-server-impl.dll"):
        assert digest(private / name.replace(".dll", "-original.dll")) == expected[name]
        shutil.copy2(BUILD / name, runtime / name)
        assert digest(runtime / name) != expected[name]

    patch = []
    source_hashes = {}
    for relative in (Path("common") / "speculative.cpp", Path("tools") / "server" / "server-context.cpp"):
        original = private / (relative.stem + "-original.cpp")
        changed = SOURCE / relative
        source_hashes[str(relative)] = {"before": digest(original), "diagnostic": digest(changed)}
        patch.extend(difflib.unified_diff(
            original.read_text(encoding="utf-8").splitlines(keepends=True),
            changed.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile="a/" + relative.as_posix(), tofile="b/" + relative.as_posix(),
        ))
    with (HERE / "diagnostic.patch").open("x", encoding="utf-8", newline="\n") as stream:
        stream.writelines(patch)

    prompt_tokens = {}
    model = reference["command"][reference["command"].index("-m") + 1]
    for row in reference["rows"]:
        case = row["case"]
        response = row["response"]
        if case not in prompt_tokens:
            run = subprocess.run(
                [str(INSTALLED / "llama-tokenize.exe"), "-m", model, "--stdin", "--ids", "--no-escape"],
                input=response["prompt"], capture_output=True, text=True, encoding="utf-8", timeout=120,
            )
            (private / f"{case}-tokenizer.log").write_text(run.stderr, encoding="utf-8")
            run.check_returncode()
            prompt_tokens[case] = json.loads(run.stdout)
        assert len(prompt_tokens[case]) == response["tokens_evaluated"]
    save(private / "prompt-tokens.json", prompt_tokens)
    manifest = {
        "installed": expected, "diagnostic": {name: digest(runtime / name) for name in expected},
        "source": source_hashes, "fixture_sha256": digest(fixture),
        "protocol_sha256": digest(HERE / "protocol.json"),
        "capture_script_sha256": digest(Path(__file__)),
        "reference_sha256": {arm: digest(reference_dir / arm / "server_U1_copy-edit.json") for arm in ("base-1", "cand-1")},
        "prompt_tokens_sha256": digest(private / "prompt-tokens.json"),
        "tokenizer_sha256": digest(INSTALLED / "llama-tokenize.exe"),
        "ngram_source_sha256": digest(SOURCE / "common" / "ngram-map.cpp"),
    }
    save(private / "before.json", manifest)

    sys.path.insert(0, str(REPO / "rtx-spark" / "bench"))
    import run_rebaseline as runner
    runner.matrix.PORT = PORT
    draft = Path(reference["command"][reference["command"].index("-md") + 1])
    shared = ["--port", str(PORT), "--spec-draft-vocab-dynamic", "16384", "--reasoning", "off"]
    for arm, extra in (
        ("mtp", []),
        ("hybrid", ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "12", "--spec-ngram-simple-size-m", "12"]),
    ):
        out = private / arm
        out.mkdir()
        runner.run_server("U1", out, 2, "copy-edit", draft_model=draft, capture_metrics=True,
                          exe=runtime / "llama-server.exe", extra_args=shared + extra)
    assert all(digest(INSTALLED / name) == value for name, value in expected.items())
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", PORT)) != 0
    print("Per-round capture complete; installed files unchanged and diagnostic port closed", flush=True)


if __name__ == "__main__":
    main()
