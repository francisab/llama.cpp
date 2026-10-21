"""Screen capacity-matched lookup against pure MTP, not the slower old hybrid."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mtp-lookup-policy-20261021"))
from capture import BUILD, INSTALLED, PORT, REPO, SOURCE, digest, save


def main():
    if not __debug__:
        raise RuntimeError("Run without -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    private = args.private.resolve(strict=True)
    assert "LLAMA_SPARK_COST_POLICY" not in os.environ and "LLAMA_TRACE" not in os.environ
    expected = json.loads((HERE.parent / "mtp-lookup-policy-20261021" / "manifest.json").read_text())["diagnostic_preflight"]["installed"]
    assert all(digest(INSTALLED / name) == value for name, value in expected.items())
    assert digest(REPO / "rtx-spark" / "bench" / "copy_edit_suite.py") == "1e527f7dc8636f4bc95f1ce952e51b6943cfc19727d41bd0bc63ead817959be9"
    assert digest(private / "llama-common-original.dll") == expected["llama-common.dll"]
    runtime = private / "runtime"
    runtime.mkdir()
    for name in expected:
        shutil.copy2(INSTALLED / name, runtime / name)
    shutil.copy2(BUILD / "llama-common.dll", runtime / "llama-common.dll")
    assert [name for name in expected if digest(runtime / name) != expected[name]] == ["llama-common.dll"]
    original = (private / "common-original.cpp").read_text(encoding="utf-8")
    candidate = (SOURCE / "common" / "common.cpp").read_text(encoding="utf-8")
    with (HERE / "candidate.patch").open("x", encoding="utf-8", newline="\n") as stream:
        stream.writelines(difflib.unified_diff(
            original.splitlines(keepends=True), candidate.splitlines(keepends=True),
            fromfile="a/common/common.cpp", tofile="b/common/common.cpp",
        ))
    save(private / "before.json", {
        "protocol_sha256": digest(HERE / "protocol.json"), "script_sha256": digest(Path(__file__)),
        "patch_sha256": digest(HERE / "candidate.patch"),
        "source_before_sha256": digest(private / "common-original.cpp"),
        "source_candidate_sha256": digest(SOURCE / "common" / "common.cpp"),
        "candidate_common_sha256": digest(runtime / "llama-common.dll"), "installed": expected,
    })
    sys.path.insert(0, str(REPO / "rtx-spark" / "bench"))
    import run_rebaseline as runner
    import compare_speed
    runner.matrix.PORT = PORT
    reference = json.loads((private.parent / "copy-edit-ngram" / "mtp" / "base-1" / "server_U1_copy-edit.json").read_text())
    draft = Path(reference["command"][reference["command"].index("-md") + 1])
    shared = ["--port", str(PORT), "--spec-draft-vocab-dynamic", "16384", "--reasoning", "off"]
    results = {}
    for arm, executable, extra in (
        ("base", INSTALLED / "llama-server.exe", []),
        ("candidate", runtime / "llama-server.exe",
         ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "12", "--spec-ngram-simple-size-m", "12"]),
    ):
        out = private / arm
        out.mkdir()
        runner.run_server("U1", out, 2, "copy-edit", draft_model=draft, capture_metrics=True,
                          exe=executable, extra_args=shared + extra)
        results[arm] = json.loads((out / "server_U1_copy-edit.json").read_text())
        assert results[arm]["prompts"] == reference["prompts"], "Frozen prompt hashes changed"
    log = (private / "candidate" / "server_U1_copy-edit.log").read_text(encoding="utf-8")
    actual = [int(value) for value in re.findall(r"lookup_snapshot target_n_rs_seq=(\d+)", log)]
    assert actual == [12], f"Unexpected actual target capacity: {actual}"
    added_memory = results["candidate"]["gpu_memory_ready_mib"] - results["base"]["gpu_memory_ready_mib"]
    assert added_memory <= 2048, f"Memory guard failed: +{added_memory} MiB"
    requests = []
    for arm, data in results.items():
        for row in data["rows"]:
            response = row["response"]
            requests.append({
                "arm": arm, "case": row["case"], "repeat": row["repeat"], "timings": response["timings"],
                "tokens_sha256": hashlib.sha256(json.dumps(response["tokens"]).encode()).hexdigest(),
                "content_sha256": response["content_sha256"],
            })
    save(HERE / "requests.json", requests)
    comparison = HERE / "comparison.json"
    assert not comparison.exists()
    code = compare_speed.main([
        str(private / "base" / "server_U1_copy-edit.json"),
        str(private / "candidate" / "server_U1_copy-edit.json"),
        "--screen", "--out", str(comparison),
    ])
    same, total = json.loads(comparison.read_text())["identical_outputs"]
    save(HERE / "checks.json", {
        "actual_target_n_rs_seq": actual, "additional_ready_gpu_mib": added_memory,
        "identical_prefixes": [same, total], "comparison_exit_code": code,
        "installed_runtime_unchanged": all(digest(INSTALLED / name) == value for name, value in expected.items()),
        "deployment_changed": False,
    })
    assert same == total == 4, "Fixed-prefix identity gate failed"
    assert all(digest(INSTALLED / name) == value for name, value in expected.items())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
