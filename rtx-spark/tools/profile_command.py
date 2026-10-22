"""Print the launch command for an explicit opt-in workload profile.

Read-only: nothing is launched, and no launcher file is read or written.
"""
import argparse
import json
from pathlib import Path, PureWindowsPath
import re
import sys

PROFILES = Path(__file__).resolve().parents[1] / "profiles" / "opt-in-profiles.json"
# Double-quoted tokens may not expand ($) or escape (`); bare tokens exclude every shell metacharacter
TOKEN = re.compile(r"""\s*(?:"([^"$`]*)"|'((?:[^']|'')*)'|([^\s"'`$;|&<>(){}#,@]+))(?=\s|$)""")
FLAG = re.compile(r"-{1,2}[A-Za-z]")
SAFE = re.compile(r"[A-Za-z0-9_./\\:=+-]+")


class ProfileError(ValueError):
    pass


def load_profiles(path=PROFILES):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {profile["id"]: profile for profile in data["profiles"]}


def split_command(text):
    text = re.sub(r"^\s*&\s+", "", re.sub(r"`\r?\n", " ", text)).strip()
    if "\n" in text:
        raise ProfileError("The file must hold one command; end continued lines with a backtick")
    tokens, position = [], 0
    while position < len(text):
        match = TOKEN.match(text, position)
        if match is None:
            raise ProfileError(f"Unsupported shell syntax near character {position}")
        double, single, bare = match.groups()
        tokens.append(single.replace("''", "'") if single is not None else double if double is not None else bare)
        position = match.end()
    if not tokens or PureWindowsPath(tokens[0]).name.lower() not in ("llama-server", "llama-server.exe"):
        raise ProfileError("The command must start with llama-server")
    return tokens


def value_index(tokens, flag):
    positions = [i for i, token in enumerate(tokens) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(tokens):
        raise ProfileError(f"Expected exactly one {flag} with a value")
    return positions[0] + 1


def check_control(profile, tokens):
    for flag, value in profile["control"]["requires"]:
        if tokens[value_index(tokens, flag)] != value:
            raise ProfileError(f"The validated control uses {flag} {value}")


def apply_edits(profile, tokens):
    tokens = list(tokens)
    for edit in profile["edits"]:
        index = value_index(tokens, edit["flag"])
        if tokens[index] != edit["from"]:
            raise ProfileError(f"Expected {edit['flag']} {edit['from']}")
        if "to" in edit:
            tokens[index] = edit["to"]
        else:
            del tokens[index - 1:index + 1]
    return tokens


def quote(token):
    return token if SAFE.fullmatch(token) else "'" + token.replace("'", "''") + "'"


def render_command(tokens):
    groups = []
    for token in tokens[1:]:
        if FLAG.match(token) or not groups:
            groups.append([token])
        else:
            groups[-1].append(token)
    executable = quote(tokens[0])
    if executable != tokens[0]:
        executable = "& " + executable
    return " `\n".join([executable] + ["  " + " ".join(quote(token) for token in group) for group in groups])


def describe_edit(edit):
    if "to" in edit:
        return f"{edit['flag']} {edit['from']} -> {edit['to']}"
    return f"remove {edit['flag']} {edit['from']}"


def render_profile(profile, text):
    tokens = split_command(text)
    check_control(profile, tokens)
    header = [f"# Opt-in profile {profile['id']}: {profile['title']}",
              "# Edits to the validated control: " + "; ".join(describe_edit(edit) for edit in profile["edits"])]
    header += [f"# Use when: {item}" for item in profile["use_when"]]
    header += [f"# Avoid when: {item}" for item in profile["avoid_when"]]
    return "\n".join(header) + "\n" + render_command(apply_edits(profile, tokens))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=PROFILES)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("list")
    actions.add_parser("show").add_argument("id")
    render = actions.add_parser("render")
    render.add_argument("id")
    render.add_argument("--command-file", type=Path, required=True, help="Text file with the validated control command")
    args = parser.parse_args(argv)
    try:
        profiles = load_profiles(args.profiles)
        if args.action == "list":
            for profile in profiles.values():
                print(f"{profile['id']}: {profile['title']}")
            return 0
        if args.id not in profiles:
            raise ProfileError(f"Unknown profile {args.id}; available: {', '.join(profiles)}")
        profile = profiles[args.id]
        if args.action == "show":
            print(json.dumps(profile, indent=2))
        else:
            print(render_profile(profile, args.command_file.read_text(encoding="utf-8")))
        return 0
    except (ProfileError, OSError, ValueError, KeyError) as error:
        print(f"profile_command: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
