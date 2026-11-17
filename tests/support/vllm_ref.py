"""vLLM's own structured-output code, run from a checkout without installing vLLM
(tests/structured/test_grammar_parity.py).

`load_text` and `load_grammar` parse vLLM's files with `ast`, keep the named top-level definitions (and, for the
masks, six methods of `StructuredOutputManager`), compile them with their file names and line numbers (a traceback
points into vLLM) and execute them in one namespace with the real llguidance and stand-ins for vLLM's imports: its
logger, `VLLMValidationError`, `is_mistral_tokenizer`, `ParserEngineReasoningAdapter` (no request here has a reasoning
parser, so no object is one). Annotations are compiled as strings (as under `from __future__ import annotations`), so
the stand-ins cover what the code runs, not its type hints. A definition vLLM renamed or moved is a LookupError, never
a silent fallback. The logger is silent: vLLM logs llguidance's matcher errors, which quote the whole grammar, i.e. the
schema.

The checkout is `LLM_MINI_INFER_VLLM_SRC`, by default a `vllm` checkout beside this repository (../vllm); `src()` is
None without one (the tests skip). From a git checkout the files are read at the tag `REV` (`git show`), whatever its
working tree holds, so a pull cannot change the oracle; a checkout without the tag is a LookupError, and a directory
that is no git checkout (an export of the tag) is read as it is.
"""

from __future__ import annotations
import __future__

import ast
import copy
import enum
import json
import linecache
import logging
import os
import subprocess
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NamedTuple

from .paths import ROOT

REFERENCE = "v0.33.0"  # the vLLM release the engine is compatible with (docs/vllm-compatibility.md)
# The release whose code the oracles run: the reference itself.
REV = "v0.33.0"
MODULE = "tests.support.vllm_ref.vllm"  # the module vLLM's definitions live in
TYPES = "vllm/v1/structured_output/backend_types.py"
GUIDANCE = "vllm/v1/structured_output/backend_guidance.py"
UTILS = "vllm/v1/structured_output/utils.py"
REQUEST = "vllm/v1/structured_output/request.py"
MANAGER = "vllm/v1/structured_output/__init__.py"
MANAGER_METHODS = ("_get_reasoner", "_fill_bitmasks", "_masked_reasoning_ends", "_get_constraint_bounds",
                   "validate_tokens", "grammar_bitmask")
# top-level names the walk over a schema and the manager read (assignments and a NamedTuple)
GUIDANCE_CONSTANTS = ("_SCHEMA_MAP_KEYWORDS", "_SUBSCHEMA_KEYWORDS")
MANAGER_NAMES = ("ConstraintBounds", "_CONSTRAINED_FROM_START", "_NO_SPEC_TOKENS")


def src() -> Path | None:
    """The vLLM checkout, or None without one."""
    path = Path(os.environ.get("LLM_MINI_INFER_VLLM_SRC") or ROOT.parent / "vllm")
    return path if all((path / f).is_file() for f in (TYPES, GUIDANCE, UTILS, REQUEST, MANAGER)) else None


class VLLMValidationError(ValueError):
    """vllm.exceptions.VLLMValidationError (a ValueError)."""


class ParserEngineReasoningAdapter:
    """vllm.parser.engine.adapters.ParserEngineReasoningAdapter: only `isinstance(reasoner, ...)` reads it, and the
    manager's `reasoner_cls` is None, so `_get_reasoner` returns None and nothing is an instance."""


def _silent_logger() -> logging.Logger:
    log = logging.getLogger("tests.support.vllm_ref")
    log.disabled = True
    log.propagate = False
    return log


def _name(root: Path, rel: str) -> str:
    """The file name vLLM's code is compiled under: the tag's lines are registered with linecache under it."""
    return f"{root / rel}@{REV}" if (root / ".git").exists() else str(root / rel)


def _source(root: Path, rel: str) -> str:
    """A vLLM file's text at `REV` (module docstring)."""
    if not (root / ".git").exists():
        return (root / rel).read_text()
    got = subprocess.run(["git", "-C", str(root), "show", f"{REV}:{rel}"], capture_output=True, text=True,
                         env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    if got.returncode:
        raise LookupError(f"the vLLM checkout {root} has no {REV}:{rel} ({got.stderr.strip()}): fetch the tag, or "
                          "point LLM_MINI_INFER_VLLM_SRC at an export of it")
    linecache.cache[_name(root, rel)] = (len(got.stdout), None, got.stdout.splitlines(True), _name(root, rel))
    return got.stdout


def _defs(root: Path, rel: str, names: tuple[str, ...], within: str = "") -> list[ast.stmt]:
    """The named definitions of a vLLM file (functions, classes, and `NAME = ...` assignments), top-level or in its
    class `within`, in the order given."""
    body = ast.parse(_source(root, rel), _name(root, rel)).body
    if within:
        cls = next((n for n in body if isinstance(n, ast.ClassDef) and n.name == within), None)
        if cls is None:
            raise LookupError(f"vLLM's {rel} has no class {within}: update tests/support/vllm_ref.py")
        body = cls.body
    found = {n.name: n for n in body if isinstance(n, ast.FunctionDef | ast.ClassDef)}
    found.update({n.targets[0].id: n for n in body if isinstance(n, ast.Assign) and len(n.targets) == 1
                  and isinstance(n.targets[0], ast.Name)})
    missing = [n for n in names if n not in found]
    if missing:
        raise LookupError(f"vLLM's {rel} has no {', '.join(missing)}{' in ' + within if within else ''}: update "
                          "tests/support/vllm_ref.py")
    return [found[n] for n in names]


def _exec(root: Path, rel: str, nodes: list[ast.stmt], ns: dict[str, Any]) -> None:
    code = compile(ast.Module(body=nodes, type_ignores=[]), _name(root, rel), "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, ns)


def load_text(root: Path) -> dict[str, Any]:
    """`get_structured_output_key` and `serialize_guidance_grammar` with what it calls: a request's schema to the
    grammar text vLLM's guidance backend compiles. No torch."""
    import llguidance

    module = ModuleType(MODULE)  # registered: dataclasses read string annotations through sys.modules
    sys.modules[MODULE] = module
    ns = module.__dict__
    ns.update({"enum": enum, "json": json, "copy": copy, "Any": Any, "llguidance": llguidance,
               "logger": _silent_logger(), "VLLMValidationError": VLLMValidationError})
    for rel, names in ((TYPES, ("StructuredOutputOptions",)),
                       (GUIDANCE, (*GUIDANCE_CONSTANTS, "_walk_json_for_additional_properties",
                                   "process_for_additional_properties", "serialize_guidance_grammar")),
                       (REQUEST, ("get_structured_output_key",))):
        _exec(root, rel, _defs(root, rel, names), ns)
    return ns


def load_grammar(root: Path) -> dict[str, Any]:
    """`load_text`, plus `GuidanceBackend` (its LLTokenizer, its grammars), `GuidanceGrammar` and the class
    `StructuredOutputManager` with only its mask methods (`manager`). Imports torch, llguidance.torch, transformers."""
    import llguidance.hf
    import numpy
    from transformers import TokenizersBackend  # the real class: `GuidanceBackend.is_tokenizer_supported` checks it

    with warnings.catch_warnings():  # its torch.compile decorator: torch.jit's deprecation warnings, at import
        warnings.simplefilter("ignore", DeprecationWarning)
        import llguidance.torch

    ns = load_text(root)
    ns.update({"os": os, "ABC": ABC, "abstractmethod": abstractmethod, "dataclass": dataclass,
               "TYPE_CHECKING": False, "llguidance_hf": llguidance.hf, "llguidance_torch": llguidance.torch,
               "is_mistral_tokenizer": lambda tokenizer: False, "TokenizersBackend": TokenizersBackend,
               "np": numpy, "NamedTuple": NamedTuple, "ParserEngineReasoningAdapter": ParserEngineReasoningAdapter})
    for rel, names in ((TYPES, ("StructuredOutputGrammar", "StructuredOutputBackend")),
                       (UTILS, ("strip_speculative_padding",)),
                       (GUIDANCE, ("GuidanceGrammar", "GuidanceBackend"))):
        _exec(root, rel, _defs(root, rel, names), ns)
    methods = _defs(root, MANAGER, MANAGER_METHODS, within="StructuredOutputManager")
    names = _defs(root, MANAGER, MANAGER_NAMES)
    cls = ast.ClassDef(name="StructuredOutputManager", bases=[], keywords=[], body=methods, decorator_list=[],
                       type_params=[], lineno=methods[0].lineno, col_offset=0)
    _exec(root, MANAGER, [*names, cls], ns)
    return ns


def backend(ns: dict[str, Any], tokenizer: Any, vocab_size: int, disable_any_whitespace: bool) -> Any:
    """vLLM's `GuidanceBackend` over a transformers tokenizer, as `--structured-outputs-config` sets it."""
    so = SimpleNamespace(disable_any_whitespace=disable_any_whitespace, disable_additional_properties=False)
    return ns["GuidanceBackend"](vllm_config=SimpleNamespace(structured_outputs_config=so), tokenizer=tokenizer,
                                 vocab_size=vocab_size)


def manager(ns: dict[str, Any], guidance: Any, k: int) -> Any:
    """vLLM's `StructuredOutputManager` (its mask methods) over `guidance`, for one request at a time with up to k
    drafts and no reasoning parser: every token is constrained."""
    import torch

    m = object.__new__(ns["StructuredOutputManager"])
    m.backend, m.reasoner_cls, m.enable_in_reasoning = guidance, None, False
    m.vllm_config = SimpleNamespace(num_speculative_tokens=k, scheduler_config=SimpleNamespace(max_num_seqs=1),
                                    model_config=SimpleNamespace(is_diffusion=False))
    m._grammar_bitmask, m._full_mask = None, torch.tensor(-1, dtype=torch.int32)
    m.fill_bitmask_parallel_threshold = 1 << 30
    return m


def request(grammar: Any) -> Any:
    """What the manager's mask methods read of a vLLM request."""
    return SimpleNamespace(use_structured_output=True,
                           structured_output_request=SimpleNamespace(grammar=grammar, reasoning_ended=None,
                                                                    masked_reasoning_end_ids=None))
