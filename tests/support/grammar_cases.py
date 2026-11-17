"""The grammar test's subset, its compiles and their child processes (tests/structured/test_grammar_test.py)."""

from __future__ import annotations

import json
import subprocess
import sys

from llm_mini_infer.api.tools import template_tool
from llm_mini_infer.grammar.spec import json_schema_grammar
from llm_mini_infer.grammar.tools import call_grammar

from .paths import ROOT

IDS = {"gemma4": {"opener": 48, "closer": 49, "quote": 52, "end": 50},
       "qwen3_coder": {"opener": 248058, "closer": 248059}}

S = {"type": "string"}


def bounds():
    """The caps' module (imported here, not at the top, so the file collects on a tree whose module holds other
    names)."""
    from llm_mini_infer.grammar import bounds

    return bounds


def child(code: str, timeout: float = 300) -> subprocess.CompletedProcess:
    """`code` in a fresh interpreter (a crash there is its exit status, not the test run's end)."""
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=timeout)


def compiled_on_threads(text: str, stack: int | None = None) -> str:
    """llguidance's check of `text`, with the token ids its rules name, on a compile thread ('' when it compiles)."""
    from llm_mini_infer.grammar.threads import CompileThreads

    from .tool_grammar_cases import check

    threads = CompileThreads(1, "test") if stack is None else CompileThreads(1, "test", stack)
    try:
        return threads.run(lambda: check(text))
    finally:
        threads.close()


def obj(props: dict, required: bool = True, **more) -> dict:
    return {"type": "object", "properties": props, **({"required": list(props)} if required else {}), **more}


def tool(params: dict, name: str = "f") -> dict:
    return template_tool({"type": "function", "function": {"name": name, "parameters": params}})


def bound(params: list[dict]) -> int:
    from llm_mini_infer.api.app import GRAMMAR_PER_BYTE, GRAMMAR_PER_TOOL, GRAMMAR_SLACK

    size = sum(len(json.dumps(p)) for p in params)
    return GRAMMAR_PER_BYTE * (size + (GRAMMAR_PER_TOOL + 1) * len(params)) + GRAMMAR_SLACK


def grammar(fam: str, params: dict, **kw) -> str:
    return call_grammar(fam, [tool(params)], "required", ids=IDS[fam], max_bytes=bound([params]), **kw)[0]


def build(mode: str, schema: dict) -> str:
    """The grammar text a `response_format` (`auto`, `guidance` compact) or a tool (a family) builds: the subset's
    check, then the text, before llguidance compiles anything."""
    if mode in ("auto", "guidance"):
        return json_schema_grammar(schema, mode, mode == "guidance")
    return grammar(mode, schema)


MODES = ["auto", "guidance", "gemma4", "qwen3_coder"]


def lark_chain(n: int) -> str:
    """A hand-written Lark grammar `n` optional rules deep."""
    return "start: r0\n" + "".join(f'r{i}: "a"? r{i + 1}\n' for i in range(n)) + f'r{n}: "b"\n'
