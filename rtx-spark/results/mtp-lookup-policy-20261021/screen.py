"""Run the one preregistered, uninstrumented cost-policy screen."""
import argparse
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from capture import BUILD, HERE, INSTALLED, PORT, REPO, SOURCE, digest, save


def main():
    if not __debug__:
        raise RuntimeError("Run without -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    private = args.private.resolve(strict=True)
    assert "LLAMA_SPARK_COST_POLICY" not in os.environ
    expected = json.loads((private / "before.json").read_text())["installed"]
    assert all(digest(INSTALLED / name) == value for name, value in expected.items())
    assert digest(SOURCE / "common" / "spark-cost-policy.h") == digest(HERE / "policy.h")
    assert digest(SOURCE / "tools" / "server" / "server-context.cpp") == digest(private / "server-context-original.cpp")
    assert "spec_cost " not in (SOURCE / "common" / "speculative.cpp").read_text(encoding="utf-8")
    runtime = private / "runtime-policy"
    runtime.mkdir()
    for name in expected:
        shutil.copy2(INSTALLED / name, runtime / name)
    shutil.copy2(BUILD / "llama-common.dll", runtime / "llama-common.dll")
    changed = [name for name in expected if digest(runtime / name) != expected[name]]
    assert changed == ["llama-common.dll"]
    original = (private / "speculative-original.cpp").read_text(encoding="utf-8")
    candidate = (SOURCE / "common" / "speculative.cpp").read_text(encoding="utf-8")
    with (HERE / "policy.patch").open("x", encoding="utf-8", newline="\n") as stream:
        stream.writelines(difflib.unified_diff(
            original.splitlines(keepends=True), candidate.splitlines(keepends=True),
            fromfile="a/common/speculative.cpp", tofile="b/common/speculative.cpp",
        ))
    save(private / "policy-before.json", {
        "protocol_sha256": digest(HERE / "policy-protocol.json"),
        "policy_sha256": digest(HERE / "policy.h"),
        "patch_sha256": digest(HERE / "policy.patch"),
        "candidate_common_sha256": digest(runtime / "llama-common.dll"),
        "installed": expected, "changed_runtime_files": changed,
        "screen_script_sha256": digest(Path(__file__)),
    })
    sys.path.insert(0, str(REPO / "rtx-spark" / "bench"))
    import run_rebaseline as runner
    import compare_speed
    runner.matrix.PORT = PORT
    reference = json.loads((private.parent / "copy-edit-ngram" / "mtp" / "base-1" / "server_U1_copy-edit.json").read_text())
    draft = Path(reference["command"][reference["command"].index("-md") + 1])
    shared = ["--port", str(PORT), "--spec-draft-vocab-dynamic", "16384", "--reasoning", "off"]
    for arm, executable, extra in (
        ("policy-base", INSTALLED / "llama-server.exe", []),
        ("policy-cand", runtime / "llama-server.exe",
         ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "12", "--spec-ngram-simple-size-m", "12"]),
    ):
        out = private / arm
        out.mkdir()
        if arm == "policy-cand":
            os.environ["LLAMA_SPARK_COST_POLICY"] = "1"
        try:
            runner.run_server("U1", out, 2, "copy-edit", draft_model=draft, capture_metrics=True,
                              exe=executable, extra_args=shared + extra)
        finally:
            os.environ.pop("LLAMA_SPARK_COST_POLICY", None)

    log = (private / "policy-cand" / "server_U1_copy-edit.log").read_text(encoding="utf-8")
    summaries = []
    for selected, bypassed, rounds, mtp, lookup in re.findall(
        r"cost_policy summary slot=0 selected=(\d+) bypassed=(\d+) mtp_rounds=(\d+) mtp_us_per_token=([\d.]+) lookup_us_per_token=([\d.]+)", log
    ):
        summaries.append({"selected_lookup": int(selected), "bypassed_lookup": int(bypassed),
                          "mtp_rounds": int(rounds), "mtp_us_per_token": float(mtp), "lookup_us_per_token": float(lookup)})
    assert len(summaries) == 6
    assert all(row["selected_lookup"] > 0 and row["bypassed_lookup"] > 0 for row in summaries)
    save(HERE / "policy-decisions.json", summaries)
    comparison = HERE / "policy-comparison.json"
    assert not comparison.exists()
    code = compare_speed.main([
        str(private / "policy-base" / "server_U1_copy-edit.json"),
        str(private / "policy-cand" / "server_U1_copy-edit.json"),
        "--screen", "--out", str(comparison),
    ])
    same, total = json.loads(comparison.read_text())["identical_outputs"]
    assert same == total == 4, "Fixed-prefix token identity gate failed"
    assert all(digest(INSTALLED / name) == value for name, value in expected.items())
    print(f"Policy screen decision exit code: {code}; installed files unchanged", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
