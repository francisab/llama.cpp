"""Held-out prompts that were not used to choose earlier RTX Spark settings."""
from pathlib import Path
import subprocess

SOURCE_COMMIT = "d80619397"
ROOT = Path(__file__).resolve().parents[2]
EN_US_CASES = ("rust256", "sqlpy256", "json256", "chat256")


def long_context(chars=64000):
    text = ""
    for path in ["docs/build.md", "docs/speculative.md"]:
        text += subprocess.check_output(
            ["git", "-C", str(ROOT), "--no-pager", "show", f"{SOURCE_COMMIT}:{path}"],
            text=True, encoding="utf-8",
        )
    if len(text) < chars:
        raise ValueError(f"Long-context source has {len(text)} characters, expected {chars}")
    return text[:chars] + "\n\nSummarize the main build and speculative decoding options described above as a list:\n"


def cases(en_us=False):
    prompts = [
        ("rust256", "Write an idiomatic Rust implementation of a bounded multi-producer multi-consumer channel "
                    "using Mutex and Condvar, with doc comments and unit tests.\n\n", 256),
        ("sqlpy256", "Write a PostgreSQL schema for an inventory system and a Python script using psycopg that "
                     "loads CSV files, validates rows, and reports anomalies.\n\n", 256),
        ("json256", "Return a JSON array of 12 fictional employees. Each object must have id, name, department, "
                    "manager_id, skills, and start_date fields.\n\n", 256),
        ("zh256", "\u8bf7\u8be6\u7ec6\u89e3\u91ca\u73b0\u4ee3\u64cd\u4f5c\u7cfb\u7edf\u4e2d\u7684\u865a\u62df"
                  "\u5185\u5b58\u662f\u5982\u4f55\u5de5\u4f5c\u7684\uff0c\u5305\u62ec\u9875\u8868\u3001TLB "
                  "\u548c\u7f3a\u9875\u5f02\u5e38\u3002\n\n", 256),
        ("es256", "Explica con detalle c\xf3mo funciona la fotos\xedntesis, desde la absorci\xf3n de luz hasta "
                  "el ciclo de Calvin.\n\n", 256),
        ("chat256", {"messages": [{"role": "user", "content":
                     "A train leaves at 14:05 and arrives at 17:50. It stops twice for 12 minutes each time. "
                     "What is the total moving time? Show brief reasoning."}]}, 256),
    ]
    if en_us:
        return [case for case in prompts if case[0] in EN_US_CASES]
    return prompts + [("long16K", long_context(), 128)]
