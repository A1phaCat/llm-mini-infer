# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/renderers/hf.py of vLLM v0.30.0 (https://github.com/vllm-project/vllm), and changed for
# llm-mini-infer.
"""Does a chat template iterate over `message.content` ("openai" parts) or print it as a string?

Vendored from vLLM's `vllm/renderers/hf.py` (`_detect_content_format` and helpers, Apache-2.0), trimmed.
The answer decides how `render.messages` shapes a message's content.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

import jinja2.nodes

from .jinja_env import make_environment

log = logging.getLogger(__name__)

ContentFormat = Literal["string", "openai"]


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    return isinstance(node, jinja2.nodes.Name) and node.ctx == "load" and node.name == varname


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    if isinstance(node, jinja2.nodes.Getitem):
        return _is_var_access(node.node, varname) and isinstance(node.arg, jinja2.nodes.Const) and node.arg.value == key
    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key
    return False


def _is_var_or_elems_access(node: jinja2.nodes.Node, varname: str, key: str | None = None) -> bool:
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(node.node, varname, key)
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)
    if isinstance(node, jinja2.nodes.Getitem) and isinstance(node.arg, jinja2.nodes.Slice):
        return _is_var_or_elems_access(node.node, varname, key)
    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _iter_nodes_assign_var_or_elems(root: jinja2.nodes.Node, varname: str) -> Iterator[tuple[jinja2.nodes.Node, str]]:
    yield root, varname
    seen = {varname}
    related = deque([varname])
    while related:
        name = related.popleft()
        for assign in root.find_all(jinja2.nodes.Assign):
            if _is_var_or_elems_access(assign.node, name) and isinstance(assign.target, jinja2.nodes.Name):
                yield assign, assign.target.name
                if assign.target.name not in seen:
                    seen.add(assign.target.name)
                    related.append(assign.target.name)


def _iter_nodes_assign_messages_item(root: jinja2.nodes.Node) -> Iterator[tuple[jinja2.nodes.Node, str]]:
    messages_varnames = [v for _, v in _iter_nodes_assign_var_or_elems(root, "messages")]
    for loop in root.find_all(jinja2.nodes.For):
        for varname in messages_varnames:
            if _is_var_or_elems_access(loop.iter, varname) and isinstance(loop.target, jinja2.nodes.Name):
                yield loop, loop.target.name
                break


def _iter_nodes_assign_content_item(root: jinja2.nodes.Node) -> Iterator[tuple[jinja2.nodes.Node, str]]:
    message_varnames = [v for _, v in _iter_nodes_assign_messages_item(root)]

    # Macro parameters that receive message.content, whatever their name.
    macro_params_by_loop: dict[int, set[str]] = {}
    loops_in_macros: set[int] = set()
    for macro in root.find_all(jinja2.nodes.Macro):
        param_names = {arg.name for arg in macro.args}
        content_params: set[str] = set()
        for call in root.find_all(jinja2.nodes.Call):
            if not (isinstance(call.node, jinja2.nodes.Name) and call.node.name == macro.name):
                continue
            for i, arg in enumerate(call.args):
                if i < len(macro.args) and any(_is_var_or_elems_access(arg, v, "content") for v in message_varnames):
                    content_params.add(macro.args[i].name)
            for kw in call.kwargs:
                if (isinstance(kw, jinja2.nodes.Keyword) and kw.key in param_names
                        and any(_is_var_or_elems_access(kw.value, v, "content") for v in message_varnames)):
                    content_params.add(kw.key)
        for loop in macro.find_all(jinja2.nodes.For):
            loops_in_macros.add(id(loop))
            if content_params:
                macro_params_by_loop[id(loop)] = content_params

    for loop in root.find_all(jinja2.nodes.For):
        if not isinstance(loop.target, jinja2.nodes.Name):
            continue
        if any(_is_var_or_elems_access(loop.iter, v, "content") for v in message_varnames):
            yield loop, loop.target.name
            continue
        params = macro_params_by_loop.get(id(loop))
        if isinstance(loop.iter, jinja2.nodes.Name) and params is not None and loop.iter.name in params:
            yield loop, loop.target.name
            continue
        if id(loop) not in loops_in_macros and isinstance(loop.iter, jinja2.nodes.Name) and loop.iter.name == "content":
            yield loop, loop.target.name


@lru_cache(maxsize=16)
def detect_content_format(source: str, default: ContentFormat = "string") -> ContentFormat:
    try:
        ast = make_environment().parse(source)
    except Exception:
        log.exception("could not parse the chat template; assuming %r content", default)
        return default
    try:
        next(_iter_nodes_assign_content_item(ast))
    except StopIteration:
        return "string"
    except Exception:
        log.exception("could not analyse the chat template; assuming %r content", default)
        return default
    return "openai"
