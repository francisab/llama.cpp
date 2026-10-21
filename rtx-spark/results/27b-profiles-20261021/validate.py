"""Validate an explicit prose profile against the actual 27B launcher recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rtx-spark" / "bench"))
import compare_speed
import en_us_quality


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_command(recipe, paths, port, depth):
    command = shlex.split(recipe["serve"]["command"].replace("`\n", "").replace("`\r\n", ""))
    if command.pop(0) != "llama-server.exe":
        raise ValueError("Unexpected launcher executable")
    replacements = {
        "-hf": None, "-hff": ("-m", paths["d27"]),
        "--mmproj-url": ("--mmproj", paths["d27_mmproj"]),
        "-hfrd": None, "--spec-draft-model": ("-md", paths["d27_dfl"]),
        "--port": ("--port", str(port)),
        "--spec-draft-n-max": ("--spec-draft-n-max", str(depth)),
    }
    resolved = []
    i = 0
    while i < len(command):
        token = command[i]
        if token in replacements:
            if i + 1 >= len(command):
                raise ValueError(f"Missing value for {token}")
            replacement = replacements[token]
            if replacement is not None:
                resolved.extend(replacement)
            i += 2
        else:
            resolved.append(token)
            i += 1
    for flag, expected in [("--batch-size", "4096"), ("--ubatch-size", "512"),
                           ("--ctx-size", "262144"), ("--spec-type", "draft-dflash")]:
        if resolved.count(flag) != 1 or resolved[resolved.index(flag) + 1] != expected:
            raise ValueError(f"Launcher {flag} differs from the frozen protocol")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["screen", "confirm", "quality"])
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9949)
    args = parser.parse_args()
    import run_rebaseline as runner

    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    catalog_bytes = args.catalog.read_bytes()
    catalog = json.loads(catalog_bytes.decode("utf-8-sig"))
    model = next(model for model in catalog["models"] if model["id"] == "qwen/qwen3.8-27b")
    recipe = next(recipe for recipe in model["recipes"] if recipe["id"] == protocol["control_recipe"])
    original = shlex.split(recipe["serve"]["command"].replace("`\n", "").replace("`\r\n", ""))
    if original[original.index("--spec-draft-n-max") + 1] != "7":
        raise ValueError("Original recipe is not DFlash7")
    runner.matrix.PORT = args.port
    runner.matrix.CONFIGS = [
        (f"P{depth} q27 launcher dflash{depth}",
         resolve_command(recipe, runner.matrix.P, args.port, depth), "decode")
        for depth in (7, 4)
    ]
    args.work.mkdir(parents=True, exist_ok=True)
    source_path = HERE / "launcher-source.json"
    source = {"catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
              "recipe": recipe["id"], "serve": recipe["serve"],
              "protocol_sha256": hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()}
    if source_path.exists():
        if json.loads(source_path.read_text(encoding="utf-8")) != source:
            raise ValueError("Launcher recipe changed during validation")
    else:
        save(source_path, source)
    if args.phase in ("confirm", "quality"):
        for pair in (1, 2):
            evidence = json.loads((HERE / f"screen-pair{pair}.json").read_text(encoding="utf-8"))
            if evidence["screen"]["decision"] != "advance":
                raise ValueError("Both speed screens must pass before longer validation")
    if args.phase == "quality":
        confirmation = json.loads((HERE / "confirm-pair1.json").read_text(encoding="utf-8"))
        if confirmation["screen"]["decision"] != "advance":
            raise ValueError("Five-repeat confirmation must pass before quality validation")
        reports = {}
        for depth in (7, 4):
            out = args.work / f"quality-{depth}"
            out.mkdir()
            runner.run_server(f"P{depth}", out, 1, "en-us-quality", allow_eos=True,
                              extra_args=["--reasoning", "off"])
            report = en_us_quality.analyze(out / f"server_P{depth}_en-us-quality.json")
            report["path"] = Path(report["path"]).name
            reports[str(depth)] = report
        passed = all(report["passed"] for report in reports.values())
        save(HERE / "quality.json", {"passed": passed, "arms": reports})
        return 0 if passed else 3
    repetitions = 2 if args.phase == "screen" else 5
    for pair in (1, 2) if args.phase == "screen" else (1,):
        outputs = {}
        for depth in (7, 4) if pair == 1 else (4, 7):
            out = args.work / f"{args.phase}-pair{pair}-{depth}"
            out.mkdir()
            runner.run_server(f"P{depth}", out, repetitions, "decode",
                              case_names=["prose256", "prose512"], capture_metrics=True)
            outputs[depth] = out / f"server_P{depth}_decode.json"
        comparison = HERE / f"{args.phase}-pair{pair}.json"
        code = compare_speed.main([str(outputs[7]), str(outputs[4]), "--screen", "--out", str(comparison)])
        evidence = json.loads(comparison.read_text(encoding="utf-8"))
        if evidence["identical_outputs"][0] != evidence["identical_outputs"][1]:
            raise RuntimeError("Draft depths did not preserve deterministic output tokens")
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
