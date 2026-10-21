"""Frozen copy-heavy editing fixtures; not a general-generation benchmark."""
import ast
import json
import re

from en_us_quality import final_text, unique_object

CODE_SOURCE = "def clamp_value(value, lower, upper):\n    return max(lower, min(value, upper))\n"
for index in range(24):
    CODE_SOURCE += (
        f"\n\ndef normalize_sensor_{index:02d}(value):\n"
        f"    adjusted = value + {index + 1}\n"
        f"    return clamp_value(adjusted, {index}, {100 + index * 7})\n"
    )
CODE_EXPECTED = CODE_SOURCE.replace("clamp_value", "bound_value")

JSON_SOURCE = {
    "schema_version": 3,
    "workers": [
        {
            "id": f"worker-{index:02d}", "queue": f"queue-{index % 4}",
            "timeout_seconds": 11 + index * 3, "max_attempts": 2 + index % 3,
            "enabled": index % 3 != 0, "labels": ["batch", f"region-{index % 5}"],
        }
        for index in range(16)
    ],
}
JSON_EXPECTED = {
    "schema_version": JSON_SOURCE["schema_version"],
    "workers": [
        {("timeout_ms" if key == "timeout_seconds" else key): (value * 1000 if key == "timeout_seconds" else value)
         for key, value in worker.items()}
        for worker in JSON_SOURCE["workers"]
    ],
}
CASE_NAMES = ("python_rename512", "json_units512")


def cases(count=512):
    prompts = [
        (CASE_NAMES[0],
         "Rename the Python function clamp_value to bound_value and update every call to it. "
         "Preserve all other code and return the complete modified file, not a patch. "
         "Return only Python code with no explanation or Markdown fences.\n\n" + CODE_SOURCE),
        (CASE_NAMES[1],
         "In every worker below, replace timeout_seconds with timeout_ms and multiply its integer value by 1000. "
         "Preserve every other key, value and array order. Return the complete modified JSON object, not a patch. "
         "Return only JSON with no explanation or Markdown fences.\n\n" + json.dumps(JSON_SOURCE, indent=2)),
    ]
    return [(name, {"messages": [{"role": "user", "content": prompt}]}, count) for name, prompt in prompts]


def score(case, response):
    try:
        text = final_text(response)
        if case == CASE_NAMES[0]:
            fence = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
            source = fence[1] if fence else text
            if ast.dump(ast.parse(source)) != ast.dump(ast.parse(CODE_EXPECTED)):
                raise ValueError("Edited Python AST differs from the exact requested rename")
            return {"passed": True, "functions": 25}
        if case == CASE_NAMES[1]:
            actual = json.loads(text, object_pairs_hook=unique_object)
            if json.dumps(actual, sort_keys=True) != json.dumps(JSON_EXPECTED, sort_keys=True):
                raise ValueError("Edited JSON differs from the exact requested unit conversion")
            return {"passed": True, "workers": 16}
        raise ValueError(f"Unknown copy-edit case: {case}")
    except (ValueError, SyntaxError, TypeError) as error:
        return {"passed": False, "error": str(error)}
