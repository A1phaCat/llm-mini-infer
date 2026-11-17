"""Tool-call grammars in each model's own call syntax, generated from the tools' schemas (architecture.md, "Tool
calls"): llguidance's Lark (llguidance 1.9.1), compiled by the engine as any grammar text.

**The shapes** (the syntax the chat templates write, and the models wrote):
- Gemma 4: `<|tool_call>call:NAME{key:value,...}<tool_call|>`, calls back to back, then `<|tool_response>`, the token
  its call turns end with (named and consumed, so a finish on the grammar's EOS cannot end the turn instead). Keys in
  the template's `dictsort` order (case-insensitive, ties in the schema's order), unquoted; strings between `<|"|>`;
  numbers, booleans and null bare; `{}` objects and `[]` arrays without spaces.
- Qwen3.8: `<tool_call>`, `\\n<function=NAME>\\n`, a `<parameter=K>\\nV\\n</parameter>\\n` block per argument in the
  schema's key order, `</function>\\n` and `</tool_call>`, calls separated by a newline, then the grammar's EOS
  (`<|im_end|>`). A string value is raw text; a boolean or null is one of the spellings the parse types so (`True`
  too); any other value is JSON (`%json`, any whitespace, as `tojson` writes it), with the `$defs` it reaches.
- `required`: one or more calls, each to one of the tools; `parallel_tool_calls` false: exactly one. A named tool:
  exactly one call to it (more would let a model that wants another tool repeat the named one until `max_tokens`, as
  Gemma 4 did). A tool without `parameters` (or `{}`) takes none (`call:NAME{}`, no parameter block).
- `response_format` with tools under `auto`: either calls, as for `required`, or the schema's JSON from the first token.
- Qwen's `auto` with a `strict` tool, or `parallel_tool_calls` false: free text, then calls (one, without parallel
  calls), the free text taking `<think>` and `</think>` too.

**Schemas.** Every tool's schema is enforced, `strict` or not, and checked at every level the same way for both families
(`_Schema.walk`). Enforced: `type` (a list too, and `nullable`), `enum`, `const`, `properties`, `required`, `items`,
`additionalProperties`, `anyOf` and `oneOf` (both as `anyOf`), `$ref` into `$defs` (or `definitions`), the root's too
(`parameters` itself must be an object schema). An object without `properties` and `additionalProperties` takes any keys
with any values (a dict); with `properties`, an absent `additionalProperties` takes no other key, and true or a schema
takes more keys after the named ones. Ignored: the annotations (`IGNORED`); an `allOf` of one schema reads as that
schema. Accepted and not enforced (logged once per grammar): `NOT_ENFORCED`. Any other keyword, an empty `enum`, `anyOf`
or `oneOf`, a type JSON does not have, a value of the wrong JSON type where a keyword takes one, a `$ref` that does not
resolve, a `$ref` cycle that can produce no value, and a schema outside the subset's merging rule and caps
(`grammar.bounds`, counted over all of a grammar's tools) are a `ToolSchemaError` (the API's 400). So is a grammar above
`max_bytes`, counted as it is built. The checks take time linear in the schemas. Names and keys are JSON-escaped
literals, so a tool's name cannot inject grammar. Special tokens (Gemma's) and Qwen's added tokens are written by id in
rules (`<[48]>`), never in terminals.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

from . import forced
from .bounds import DEFS, MAX_NESTING, BoundError, Caps, deepest, def_key, json_depth, keep_apart

log = logging.getLogger(__name__)

IGNORED = frozenset(("title", "description", "default", "examples", "$comment", "deprecated", "$schema", "$id",
                     "readOnly", "writeOnly", "discriminator"))
NOT_ENFORCED = ("pattern", "format", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
                "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "uniqueItems")
STRUCTURE = frozenset(("type", "nullable", "enum", "const", "properties", "required", "items", "additionalProperties",
                       "anyOf", "oneOf", "allOf", "$ref", "$defs", "definitions"))
KNOWN = IGNORED | STRUCTURE | frozenset(NOT_ENFORCED)
JSON_TYPES = frozenset(("string", "integer", "number", "boolean", "null", "object", "array"))
_INT = "/-?[0-9]+/"
_NUM = "/-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?/"
_TEXT = "/(.|\\n)*/"
# A Qwen parameter's boolean or null, top level: what the parse types as one (`output.tool_args.coerce`, which compares
# case-insensitively), as the models write them (Qwen3.8 wrote `True`)
_SPELLINGS = {"boolean": ("true", "false", "True", "False", "TRUE", "FALSE"), "null": ("null", "Null", "NULL")}


class ToolSchemaError(ValueError):
    """A tool's schema the call grammar cannot take (a 400, param `tools`)."""


def _lit(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def _key_order(keys: Any) -> list:
    """Jinja's `dictsort` order, which Gemma's template writes keys in: case-insensitive, ties in the given order."""
    return sorted(keys, key=lambda k: str(k).lower())


def _open(s: dict) -> bool:
    """An object schema without `properties` and `additionalProperties`: any keys with any values."""
    return "properties" not in s and "additionalProperties" not in s


def _map_refs(s: Any, fn: Any, close: bool = False) -> Any:
    """A copy of schema `s` with each `$ref` passed through `fn`, in schema positions only (not in `const` or `enum`
    values); `close`: an object with `properties` and no `additionalProperties` gets it false (JSON Schema's default
    is true; the grammars' rule is no other key)."""
    if not isinstance(s, dict):
        return s
    out = {}
    for k, v in s.items():
        if k == "$ref" and isinstance(v, str):
            out[k] = fn(v)
        elif k == "properties" and isinstance(v, dict):
            out[k] = {p: _map_refs(x, fn, close) for p, x in v.items()}
        elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
            out[k] = [_map_refs(x, fn, close) for x in v]
        elif k in ("items", "additionalProperties") and isinstance(v, dict):
            out[k] = _map_refs(v, fn, close)
        else:
            out[k] = v
    if close and "properties" in out and "additionalProperties" not in out:
        out["additionalProperties"] = False
    return out


def _refs_in(s: Any) -> list[str]:
    """The `$ref`s in schema `s`, in schema positions."""
    found: list[str] = []
    _map_refs(s, lambda ref: found.append(ref) or ref)
    return found


def _dumped_size(x: Any, memo: dict[int, int]) -> int:
    """At most the length of `json.dumps(x, ensure_ascii=False)` after `_map_refs`' `_to_defs` (Qwen's `%json` text
    without what it adds): each value the document holds measured once (`memo`, by identity), and counted at each of
    its uses, so a schema shared by many alternatives costs its size once in time. A walk with its own stack: no Python
    frame a level (a decoded JSON document holds no cycle)."""
    todo: list[tuple[Any, bool]] = [(x, False)]
    while todo:
        y, ready = todo.pop()
        if id(y) in memo:
            continue
        if not isinstance(y, dict | list):
            memo[id(y)] = len(json.dumps(y, ensure_ascii=False))
        elif not ready:  # its values first, then itself
            todo.append((y, True))
            todo += ((v, False) for v in (y.values() if isinstance(y, dict) else y))
        elif isinstance(y, dict):
            got = 2 * len(y) + sum(len(json.dumps(k, ensure_ascii=False)) + 2 + memo[id(v)] for k, v in y.items())
            ref = y.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/definitions/"):
                got -= len("definitions") - len("$defs")
            memo[id(y)] = got
        else:
            memo[id(y)] = 2 * len(y) + sum(memo[id(v)] for v in y)
    return memo[id(x)]


def _to_defs(ref: str) -> str:
    """A draft-07 `#/definitions/X` ref as `#/$defs/X` (the definitions Qwen's `%json` items carry)."""
    return "#/$defs/" + ref.removeprefix("#/definitions/") if ref.startswith("#/definitions/") else ref


class _Rules:
    """The grammar's rules, deduplicated by body; `defs` resolve `$ref`s per tool. `max_bytes`: the most text the rules
    may take, counted as they are added (`ToolSchemaError` the moment it is passed)."""

    def __init__(self, max_bytes: int | None = None) -> None:
        self.bodies: dict[str, str] = {}  # body -> name
        self.order: list[tuple[str, str]] = []
        self.at: dict[str, int] = {}  # a reserved rule's name -> its place in `order`
        self.refs: dict[tuple[int, str], str] = {}
        self.loose: set[str] = set()  # keywords accepted and not enforced
        self.max_bytes = max_bytes
        self.size = 0  # the rules' text so far, in characters (at most its bytes)

    def grow(self, n: int) -> None:
        self.size += n
        if self.max_bytes is not None and self.size > self.max_bytes:
            raise ToolSchemaError(f"the tools' call grammar passes {self.max_bytes} bytes, the most this server builds "
                                  "for their schemas")

    def room(self) -> int | None:
        """The text the rules may still take (None: no bound)."""
        return None if self.max_bytes is None else self.max_bytes - self.size

    def add(self, body: str, prefix: str = "r", attrs: str = "") -> str:
        key = attrs + "|" + body
        name = self.bodies.get(key)
        if name is None:
            name = f"{prefix}_{len(self.order)}"
            self.grow(len(name) + len(attrs) + len(body) + 3)
            self.bodies[key] = name
            self.order.append((name + attrs, body))
        return name

    def reserve(self, prefix: str) -> str:
        name = f"{prefix}_{len(self.order)}"
        self.grow(len(name) + 3)
        self.at[name] = len(self.order)
        self.order.append((name, ""))
        return name

    def set(self, name: str, body: str) -> None:
        self.grow(len(body))
        self.order[self.at[name]] = (name, body)

    def text(self) -> str:
        return "".join(f"{n}: {b}\n" for n, b in self.order)


class _Schema:
    """One tool's schema: its checks (`walk`, at every level) and `$ref` resolution."""

    def __init__(self, rules: _Rules, root: dict, tool: int, name: str, caps: Caps):
        self.rules, self.root, self.tool, self.name, self.caps = rules, root, tool, name, caps
        for k, v in (("$defs", root.get("$defs")), ("definitions", root.get("definitions"))):
            if v is not None and not isinstance(v, dict):
                raise ToolSchemaError(f"tool {name!r}: '{k}' must be an object")
        self.defs = {**(root.get("definitions") or {}), **(root.get("$defs") or {})}
        caps.defined(len(self.defs))
        self.place = {k: i for i, k in enumerate(self.defs)}  # declaration order
        self._text: dict[str, str] = {}  # a definition as Qwen's `%json` items carry it
        self._refs: dict[str, list[str]] = {}  # the definitions a definition's `$ref`s name
        self._chains: dict[str, tuple] = {}  # `alias_chain`'s, by its first definition
        self._open_chains: dict[str, tuple] = {}  # the same, the chains that end (another may join one)

    def check(self, s: Any, where: str) -> dict:
        """Schema `s` as the grammar reads it: its keywords known (the loose ones logged), the merging rule kept
        (`bounds.keep_apart`), a one-member `allOf` read as that member beside the annotations."""
        while True:
            if s is True or s == {}:
                return {}
            if not isinstance(s, dict):
                raise ToolSchemaError(f"tool {self.name!r}: the schema at {where} must be an object")
            for k in s:
                if k not in KNOWN:
                    raise ToolSchemaError(f"tool {self.name!r}: the keyword {k!r} at {where} is not supported in a "
                                          "tool's schema for a constrained call")
                if k in NOT_ENFORCED:
                    self.rules.loose.add(k)
            try:
                keep_apart(s, where, IGNORED.__contains__)
            except BoundError as exc:
                raise ToolSchemaError(f"tool {self.name!r}: {exc}") from None
            if "allOf" not in s:
                return s
            member = s["allOf"][0]
            if member is not True and not isinstance(member, dict):
                raise ToolSchemaError(f"tool {self.name!r}: the schema at {where}/allOf/0 must be an object")
            s = {**{k: v for k, v in s.items() if k != "allOf"}, **(member if member is not True else {})}

    def resolve(self, ref: str, where: str) -> tuple[str, dict]:
        key = self.key_of(ref)
        if key is None:
            raise ToolSchemaError(f"tool {self.name!r}: the $ref {ref!r} at {where} does not resolve")
        return key, self.defs[key]

    def walk(self, s: Any, where: str) -> None:
        """Every schema `s` reaches, checked once, depth first in the document's order: its keywords (the loose ones
        logged), a non-empty `enum`, `anyOf` and `oneOf`, `type` a JSON type or a list of them, `required` a list of
        names, `properties` an object of schemas, `items` and `additionalProperties` schemas (or booleans), each `$ref`
        resolved (its target walked once), a nested `$defs` checked as definitions no `$ref` reaches, an `$id` at the
        root only; the caps' counts (`bounds.Caps`) and the nesting (`bounds.deepest`); then the `$ref`s that can
        produce no value (`_productive`) are refused."""
        refs: dict[str, tuple[dict, str]] = {}
        height: dict[Any, int] = {None: 0}  # each definition's depth (None: the parameters'), its own levels
        uses: dict[Any, dict[str, int]] = {}  # the definitions each one's $refs name, at the deepest level
        stack: list[tuple[Any, str, Any, int]] = [(s, where, None, 1)]  # a schema, where, its definition, level
        while stack:
            raw, at, owner, level = stack.pop()
            x = self.check(raw, at)
            height[owner] = max(height[owner], level)
            self.caps.add(x, at)
            if "$id" in x and raw is not s:
                raise ToolSchemaError(f"tool {self.name!r}: the '$id' at {at} is not supported: only the root's is")
            todo: list[tuple[Any, str, Any, int]] = []
            node, here = raw, at
            while isinstance(node, dict):  # a nested $defs, beside an allOf or in its member: what the grammar may
                for k in DEFS:  # carry (Qwen's items), checked and counted
                    if node is s or k not in node:
                        continue
                    if not isinstance(node[k], dict):
                        raise ToolSchemaError(f"tool {self.name!r}: '{k}' at {here} must be an object")
                    self.caps.defined(len(node[k]))
                    for name, d in node[k].items():
                        nested = (here, k, name)  # no $ref reaches it: its own depth, apart from the definitions'
                        height[nested] = 0
                        todo.append((d, f"{here}/{k}/{name}", nested, 1))
                member = node.get("allOf")
                node = member[0] if isinstance(member, list) and member else None
                here += "/allOf/0"
            if "$ref" in x:
                key, target = self.resolve(x["$ref"], at)
                named = uses.setdefault(owner, {})
                named[key] = max(named.get(key, 0), level)
                if key not in refs:
                    refs[key] = (target, x["$ref"])
                    height[key] = 0
                    todo.append((target, f"#/$defs/{key}", key, 1))
            for alt in ("anyOf", "oneOf"):
                if alt in x:
                    subs = x[alt]
                    if not isinstance(subs, list) or not subs:
                        raise ToolSchemaError(f"tool {self.name!r}: '{alt}' at {at} must be a non-empty list")
                    todo += [(y, f"{at}/{alt}/{i}", owner, level + 1) for i, y in enumerate(subs)]
            if "enum" in x and (not isinstance(x["enum"], list) or not x["enum"]):
                raise ToolSchemaError(f"tool {self.name!r}: 'enum' at {at} must be a non-empty list")
            t = x.get("type")
            if t is not None and not isinstance(t, str) and not (isinstance(t, list) and all(isinstance(m, str)
                                                                                           for m in t)):
                raise ToolSchemaError(f"tool {self.name!r}: 'type' at {at} must be a string or a list of strings")
            for t in _types(x):
                if t not in JSON_TYPES:
                    raise ToolSchemaError(f"tool {self.name!r}: the type {t!r} at {at} is not a JSON type")
            req = x.get("required")
            if req is not None and not (isinstance(req, list) and all(isinstance(m, str) for m in req)):
                raise ToolSchemaError(f"tool {self.name!r}: 'required' at {at} must be a list of strings")
            props = x.get("properties")
            if props is not None:
                if not isinstance(props, dict):
                    raise ToolSchemaError(f"tool {self.name!r}: 'properties' at {at} must be an object")
                todo += [(v, f"{at}/properties/{k}", owner, level + 1) for k, v in props.items()]
            for k, sub in (("items", "items"), ("additionalProperties", "additional")):
                v = x.get(k)
                if v is None or v is True or (v is False and k == "additionalProperties"):
                    continue
                if not isinstance(v, dict):
                    raise ToolSchemaError(f"tool {self.name!r}: '{k}' at {at} must be a schema")
                todo.append((v, f"{at}/{sub}", owner, level + 1))
            stack += reversed(todo)
        depth = deepest(height, uses)
        if depth > MAX_NESTING:
            raise ToolSchemaError(f"tool {self.name!r}: its schema nests {depth} levels deep, counting its $refs' "
                                  f"definitions where they are used, more than {MAX_NESTING}")
        self._productive(refs)

    def object_root(self, s: dict) -> dict:
        """The tool's `parameters` as the call's object: a root `$ref` (pydantic's recursive model) followed;
        anything but an object schema (a type, alternatives, values) refused."""
        s = self.check(s, "#")
        followed: set[str] = set()
        while "$ref" in s:
            key, target = self.resolve(s["$ref"], "#")
            if key in followed:  # refused by `_productive` first
                raise ToolSchemaError(f"tool {self.name!r}: the $ref {s['$ref']!r} at # is a cycle")
            followed.add(key)
            s = self.check(target, f"#/$defs/{key}")
        if _types(s) not in ([], ["object"]) or any(k in s for k in ("anyOf", "oneOf", "enum", "const")):
            raise ToolSchemaError(f"tool {self.name!r}: its parameters must be an object schema")
        return s

    def _productive(self, refs: dict[str, tuple[dict, str]]) -> None:
        """A `$ref` whose definition can produce no value (a cycle through refs alone, or through required keys) is
        refused. Whether a schema can produce a value is an AND (its object's required keys) and OR (its alternatives)
        of whether the definitions it refers to can (`_Produces`); the definitions that can are the least fixpoint,
        found by propagation from those that can outright, in time linear in the schemas."""
        p = _Produces(self)
        roots = {k: p.build(target) for k, (target, _) in refs.items()}
        good = p.solve(roots)
        for k in refs:
            if k not in good:
                raise ToolSchemaError(f"tool {self.name!r}: the $ref {refs[k][1]!r} is a cycle that can produce no "
                                      "value")

    def key_of(self, ref: Any) -> str | None:
        """The definition a `$ref` names, if it resolves (`bounds.def_key`)."""
        return def_key(ref, self.defs)

    def def_text(self, key: str) -> str:
        """Definition `key` as Qwen's `%json` items carry it (`_Qwen._json`), made once."""
        got = self._text.get(key)
        if got is None:
            got = self._text[key] = json.dumps(_map_refs(self.defs[key], _to_defs, close=True), ensure_ascii=False)
        return got

    def def_refs(self, key: str) -> list[str]:
        """The definitions definition `key`'s `$ref`s name, found once."""
        got = self._refs.get(key)
        if got is None:
            got = self._refs[key] = [k for k in map(self.key_of, _refs_in(self.defs[key])) if k is not None]
        return got

    def alias_chain(self, key: str) -> tuple[dict, dict, frozenset, str, bool]:
        """Definition `key` followed through the `$ref`s it is (Qwen's value walk, `_Qwen._value`): the schema at the
        chain's end with each definition's keywords on the way under the next one's, the same keywords without the
        last definition's (`prefix`), the chain's definitions, its last one, and whether it closes a cycle (its last
        `$ref` names one already on it). Made once a definition; chains that end share their tails."""
        got = self._chains.get(key)
        if got is not None:
            return got
        path, seen = [key], {key}
        while True:
            t = self.check(self.defs[path[-1]], f"#/$defs/{path[-1]}")
            nxt = self.key_of(t.get("$ref")) if "$ref" in t else None
            if nxt is None or nxt in seen or nxt in self._open_chains:
                break
            path.append(nxt)
            seen.add(nxt)
        closes = nxt is not None and nxt in seen
        if nxt is not None and not closes:  # joins a chain that ends
            view, prefix, keys, last, _ = self._open_chains[nxt]
            own = {k: v for k, v in self.check(self.defs[path[-1]], "").items() if k != "$ref"}
            view, prefix, keys = {**own, **view}, {**own, **prefix}, keys | {path[-1]}
        else:
            last = path[-1]
            view, prefix, keys = self.check(self.defs[last], ""), {}, frozenset((last,))
        for i in range(len(path) - 1, -1, -1):
            k = path[i]
            if i < len(path) - 1:
                own = {a: b for a, b in self.check(self.defs[k], "").items() if a != "$ref"}
                view, prefix, keys = {**own, **view}, {**own, **prefix}, keys | {k}
            got = (view, prefix, keys, last, closes)
            if i == 0:
                self._chains[k] = got
            if not closes:
                self._open_chains[k] = self._chains[k] = got
        return self._chains[key]

    def reachable(self, s: Any, limit: int | None = None) -> list[str]:
        """The definitions `s` reaches through `$ref`s, transitively, in their declaration order. `limit`: the most
        text their `%json` copies may take, counted as they are found (past it, the grammar bound's
        `ToolSchemaError`)."""
        seen: set[str] = set()
        todo = [k for k in map(self.key_of, _refs_in(s)) if k is not None]
        size = 0
        while todo:
            key = todo.pop()
            if key in seen:
                continue
            seen.add(key)
            size += len(key) + len(self.def_text(key)) + 6
            if limit is not None and size > limit:
                raise ToolSchemaError(f"the tools' call grammar passes {self.rules.max_bytes} bytes, the most this "
                                      "server builds for their schemas")
            todo += self.def_refs(key)
        return sorted(seen, key=self.place.__getitem__)


_TRUE, _FALSE = -1, -2  # `_Produces`' constant nodes


class _Produces:
    """Whether schemas can produce a value, as a graph: a node is an AND (an object's required keys) or an OR (its
    alternatives, or a definition: its schema), its children nodes or the constants; one node per schema the document
    holds, so a shape that alternatives share is built once. A schema's own rule (`_Schema.walk`'s):
    anything not a dict, `{}`, `const`, `enum` and a non-object type can; a `$ref`, if its definition can; `anyOf` (or
    `oneOf`), if one alternative can; an object, if each required key it names can."""

    def __init__(self, w: _Schema):
        self.w = w
        self.kind: list[bool] = []  # True: AND
        self.kids: list[list[int]] = []
        self.var: dict[str, int] = {}  # a definition -> its node
        self.memo: dict[int, tuple[Any, int]] = {}  # id(a document's schema) -> (it, its node)

    def node(self, is_and: bool, kids: list[int]) -> int:
        self.kind.append(is_and)
        self.kids.append(kids)
        return len(self.kind) - 1

    def ref(self, key: str) -> int:
        n = self.var.get(key)
        if n is None:
            n = self.var[key] = self.node(False, [])  # its definition's node is linked by `solve`
        return n

    def build(self, s: Any) -> int:
        """The node of a schema the document holds (built once)."""
        got = self.memo.get(id(s))
        if got is not None and got[0] is s:
            return got[1]
        n = self.own(s)
        self.memo[id(s)] = (s, n)
        return n

    def own(self, s: Any) -> int:
        if not isinstance(s, dict) or not s:
            return _TRUE
        s = self.w.check(s, "")  # a one-member allOf as its member
        if "$ref" in s:
            key = self.w.key_of(s["$ref"])
            return _FALSE if key is None else self.ref(key)
        for alt in ("anyOf", "oneOf"):
            if alt in s:
                kids = []
                for x in s[alt]:
                    n = self.build(x)
                    if n == _TRUE:
                        return _TRUE
                    if n != _FALSE:
                        kids.append(n)
                return self.node(False, kids) if kids else _FALSE
        if "const" in s or "enum" in s:
            return _TRUE
        types = _types(s) or (["object"] if "properties" in s else ["array"] if "items" in s else [])
        if not types or any(t != "object" for t in types):
            return _TRUE
        props = s.get("properties") or {}
        kids = []
        for k in s.get("required") or ():
            if k in props:
                n = self.build(props[k])
                if n == _FALSE:
                    return _FALSE
                if n != _TRUE:
                    kids.append(n)
        return self.node(True, kids) if kids else _TRUE

    def solve(self, roots: dict[str, int]) -> set[str]:
        """The definitions that can produce a value, given each one's schema node (`roots`): from those that can
        outright, each node that becomes true counts down its parents (an AND waits for all its children, an OR for
        one)."""
        for key, root in roots.items():
            n = self.ref(key)
            if root == _TRUE:
                self.kids[n] = []
                self.kind[n] = True  # an AND of nothing: true
            elif root != _FALSE:
                self.kids[n] = [root]
        parents: list[list[int]] = [[] for _ in self.kind]
        need = [0] * len(self.kind)
        todo = []
        for i, kids in enumerate(self.kids):
            for c in kids:
                parents[c].append(i)
            need[i] = len(kids) if self.kind[i] else 1
            if self.kind[i] and not kids:
                todo.append(i)
        true = [False] * len(self.kind)
        for i in todo:
            true[i] = True
        while todo:
            for p in parents[todo.pop()]:
                if not true[p]:
                    need[p] -= 1
                    if need[p] <= 0:
                        true[p] = True
                        todo.append(p)
        return {k for k, n in self.var.items() if true[n]}


def _target(w: _Schema, ref: str, where: str) -> dict:
    """The definition `ref` names in `w`'s schema (`forced.Runs`' resolve)."""
    return w.resolve(ref, where)[1]


def _fixed_run(family: str, w: _Schema, root: dict | None, params: Any, q: _Qwen | None, ids: dict[str, int],
               last: bool) -> Any:
    """A tool's call, as `grammar.forced` counts its fixed output (`last`: the end token follows the call)."""
    resolve = functools.partial(_target, w)
    if family == "gemma4":
        runs = None
        if root is not None:
            runs = forced.Runs(params, forced.Syntax(quote=b"\xff" + f"[{ids['quote']}]".encode()), resolve)
            runs.definitions(params, "#", None)
        return forced.gemma_call(runs, root, w.name, ids, last)
    items = q.runs(w, root, params, resolve) if root is not None and q is not None else []
    more = root is not None and (_open(root) or root.get("additionalProperties", False) is not False)
    return forced.qwen_call(items, more, w.name, ids)


def _types(s: dict) -> list[str]:
    t = s.get("type")
    types = [t] if isinstance(t, str) else list(t) if isinstance(t, list) else []
    if s.get("nullable") is True and "null" not in types:
        types.append("null")
    return types


# -- Gemma 4 --------------------------------------------------------------------------------------------------------

def _gemma_literal(v: Any, q: str) -> str:
    """A value as Gemma's template writes it, as a sequence of grammar items."""
    if isinstance(v, str):
        return f"{q} {_lit(v)} {q}" if v else f"{q} {q}"
    if isinstance(v, bool):
        return _lit("true" if v else "false")
    if v is None:
        return _lit("null")
    if isinstance(v, int | float):
        return _lit(json.dumps(v))
    if isinstance(v, list):
        inner = ' "," '.join(_gemma_literal(x, q) for x in v)
        return f'"[" {inner} "]"' if v else '"[]"'
    if isinstance(v, dict):
        inner = ' "," '.join(f"{_lit(str(k) + ':')} {_gemma_literal(v[k], q)}" for k in _key_order(v))
        return f'"{{" {inner} "}}"' if v else '"{}"'
    raise ToolSchemaError(f"an enum or const value of type {type(v).__name__}")


class _Gemma:
    def __init__(self, rules: _Rules, quote: str):
        self.rules, self.q = rules, quote
        self._any: str | None = None
        self._memo: dict[int, tuple[Any, str]] = {}  # id(a schema the document holds) -> (it, its rule)

    def any_key(self) -> str:
        """A key of an object the schema does not name: unquoted, as the template writes it, or between `<|"|>` (the
        parse takes both, up to the first `:`)."""
        return self.rules.add(f'/[^:,{{}}\\[\\]]+/ ":" | {self.q} /[^:]*/ {self.q} ":"', "anykey")

    def any_value(self) -> str:
        if self._any is None:
            r = self.rules
            self._any = r.reserve("any")
            key = self.any_key()
            obj = r.add(f'"{{}}" | "{{" {key} {self._any} ("," {key} {self._any})* "}}"', "anyobj")
            arr = r.add(f'"[]" | "[" {self._any} ("," {self._any})* "]"', "anyarr")
            s = self.string()
            r.set(self._any, f'{s} | {_NUM} | "true" | "false" | "null" | {obj} | {arr}')
        return self._any

    def string(self) -> str:
        return self.rules.add(f"{self.q} {_TEXT} {self.q}", "str")

    def value(self, w: _Schema, s: Any, where: str) -> str:
        """A value of schema `s` (one the document holds: its rule is built once, however many alternatives reach
        it)."""
        got = self._memo.get(id(s))
        if got is not None and got[0] is s:
            return got[1]
        name = self._value(w, s, where)
        self._memo[id(s)] = (s, name)
        return name

    def _value(self, w: _Schema, s: Any, where: str) -> str:
        s = w.check(s, where)
        r = self.rules
        if "$ref" in s:
            key, target = w.resolve(s["$ref"], where)
            got = r.refs.get((w.tool, key))
            if got is None:
                got = r.refs[(w.tool, key)] = r.reserve("def")
                r.set(got, self.value(w, target, f"#/$defs/{key}"))
            return got
        for alt in ("anyOf", "oneOf"):
            if alt in s:
                return r.add(" | ".join(self.value(w, x, f"{where}/{alt}/{i}") for i, x in enumerate(s[alt])), "alt")
        if "const" in s:
            return r.add(_gemma_literal(s["const"], self.q), "const")
        if "enum" in s:
            return r.add(" | ".join(_gemma_literal(v, self.q) for v in s["enum"]), "enum")
        types = _types(s)
        if not types:
            types = ["object"] if "properties" in s else ["array"] if "items" in s else []
        if not types:
            return self.any_value()
        alts = []
        for t in types:
            if t == "string":
                alts.append(self.string())
            elif t == "integer":
                alts.append(_INT)
            elif t == "number":
                alts.append(_NUM)
            elif t == "boolean":
                alts.append('"true" | "false"')
            elif t == "null":
                alts.append('"null"')
            elif t == "object":
                alts.append(self.obj(w, s, where))
            else:  # array (`_Schema.walk` refused any other type)
                items = s.get("items")
                item = self.value(w, items, f"{where}/items") if items not in (None, True) else self.any_value()
                alts.append(r.add(f'"[]" | "[" {item} ("," {item})* "]"', "arr"))
        return r.add(" | ".join(alts), "v")

    def obj(self, w: _Schema, s: dict, where: str) -> str:
        """An object (with its braces): the named keys in the template's order, the required ones present, then more
        keys when `additionalProperties` allows them; any keys with any values without either (`_open`)."""
        r = self.rules
        if _open(s):
            return r.add(f'"{{}}" | "{{" {self.any_key()} {self.any_value()} ("," {self.any_key()} '
                         f'{self.any_value()})* "}}"', "anyobj")
        props = s.get("properties") or {}
        required = set(s.get("required") or ())
        addl = s.get("additionalProperties", False)
        extra = None
        if addl is not False:
            extra = r.add(f"{self.any_key()} "
                          f"{self.any_value() if addl is True else self.value(w, addl, where + '/additional')}",
                          "extra")
        items = [(k, self.value(w, props[k], f"{where}/properties/{k}"), k in required) for k in _key_order(props)]
        # a_i: the items from i on, each after a comma; b_i: the same with no comma before the first
        a = f'("," {extra})*' if extra else ""
        b = f"({extra} (\",\" {extra})*)?" if extra else ""
        for k, v, req in reversed(items):
            pair = f"{_lit(k + ':')} {v}"
            if req:
                a, b = f'"," {pair} {a}'.strip(), f"{pair} {a}".strip()
            else:
                a_new = f'("," {pair})? {a}'.strip()
                b = f"({pair} {a} | {b})" if b else f"({pair} {a})?".strip()
                a = a_new
            a = r.add(a or '""', "seq")
            b = r.add(b or '""', "first")
        body = b or '""'
        return r.add(f'"{{" {body} "}}"', "obj")


# -- Qwen3.8 --------------------------------------------------------------------------------------------------------

_ATOMS = (str, int, float, bool, type(None))


def _shape(s: Any) -> tuple:
    """A key that two schemas share when they have the same keys and values, a container value being the same object
    (one the document holds, alive while the grammar is built): equal keys mean equal schemas."""
    def atom(v: Any) -> tuple:
        return (type(v).__name__, v) if isinstance(v, _ATOMS) else ("id", id(v))

    return tuple((k, *atom(v)) for k, v in s.items()) if isinstance(s, dict) else ("value", *atom(s))


def _content(s: dict, memo: dict[int, tuple[Any, str]]) -> tuple:
    """A key equal for two schemas exactly when their `json.dumps(sort_keys=True)` are, each container value dumped
    once (`memo`, by identity): annotations beside many `$ref`s cost their size once."""
    def one(v: Any) -> tuple:
        if isinstance(v, str | int | type(None)):  # bool too, by its type's name
            return (type(v).__name__, v)
        if isinstance(v, float):
            return ("float", repr(v))  # -0.0 is not 0.0
        got = memo.get(id(v))
        if got is None or got[0] is not v:
            got = memo[id(v)] = (v, json.dumps(v, sort_keys=True, default=str))
        return ("json", got[1])

    return tuple(sorted((k, *one(v)) for k, v in s.items()))


class _Qwen:
    def __init__(self, rules: _Rules):
        self.rules = rules
        self._items: dict[str, str] = {}  # a sub-schema's own JSON -> its `%json` item
        self._values: dict[tuple, tuple[bool, list[str], str | None]] = {}  # `_shape(schema)` -> `_value`'s result
        self._dumped: dict[int, tuple[Any, str]] = {}  # `_content`'s

    def _json(self, w: _Schema, s: dict) -> str:
        """A `%json` item: the sub-schema with the definitions it reaches (`definitions` refs as `$defs` ones), its
        objects with `properties` closed as the module docstring says. Each definition's text is made once
        (`_Schema.def_text`); the item's size is estimated against the grammar's bound before it is copied
        (`_dumped_size`), and counted before it is joined."""
        room = self.rules.room()
        if room is not None and _dumped_size({k: v for k, v in s.items() if k != "$defs"}, {}) > room:
            raise ToolSchemaError(f"the tools' call grammar passes {self.rules.max_bytes} bytes, the most this server "
                                  "builds for their schemas")  # before anything is copied: alternatives share parts
        sub = _map_refs(s, _to_defs, close=True)
        own = json.dumps(sub, ensure_ascii=False)
        got = self._items.get(own)
        if got is not None:
            return got
        if room is not None and isinstance(sub.get("$defs"), dict):  # replaced by the reached definitions, if any
            room += len(json.dumps(sub["$defs"], ensure_ascii=False))
        keys = w.reachable(s, None if room is None else room - len(own))
        parts = []  # json.dumps's text of the sub-schema with its `$defs`, then its `x-guidance`, set
        for k, v in sub.items():
            if k == "$defs" and keys:
                text = self._defs(w, keys)
            elif k == "x-guidance":
                text = json.dumps({**(v or {}), "whitespace_flexible": True}, ensure_ascii=False)
            else:
                text = json.dumps(v, ensure_ascii=False)
            parts.append(f"{json.dumps(k, ensure_ascii=False)}: {text}")
        if keys and "$defs" not in sub:
            parts.append('"$defs": ' + self._defs(w, keys))
        if "x-guidance" not in sub:
            parts.append('"x-guidance": {"whitespace_flexible": true}')
        got = self._items[own] = "%json {" + ", ".join(parts) + "}"
        return got

    @staticmethod
    def _defs(w: _Schema, keys: list[str]) -> str:
        return "{" + ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {w.def_text(k)}" for k in keys) + "}"

    def value_rule(self, w: _Schema, k: str, s: Any, where: str) -> str:
        """A `<parameter=K>` block: a raw text body for a string, enum or const strings, JSON for the rest; the
        value's rules made once per parameter schema (`_shape`), however many parameters share it."""
        r = self.rules
        head, tail = _lit(f"<parameter={k}>\n"), _lit("\n</parameter>\n")
        key = _shape(s)
        got = self._values.get(key)
        if got is None:
            got = self._values[key] = [*self._value(w, s, where), None]
        raw, texts, body, json_rule = got
        alts: list[str] = []
        if raw:
            alts.append(r.add(f"{head} {_TEXT} {tail}", "qtext", "[lazy]"))
        if texts:
            alts.append(r.add(f"{head} ({' | '.join(_lit(t) for t in texts)}) {tail}", "qenum"))
        if body is not None:  # a rule of its own: parameters with the same schema share it
            if json_rule is None:
                json_rule = got[3] = r.add(self._json(w, body), "json")
            alts.append(r.add(f"{head} {json_rule} {tail}", "qjson"))
        return r.add(" | ".join(alts), "qparam")

    def _value(self, w: _Schema, s: Any, where: str) -> tuple[bool, list[str], dict | None]:
        """A parameter's value: whether it takes raw text, the literal raw values (each once), and the schema of its
        JSON values."""
        texts: list[str] = []  # literal raw values
        raw = False
        json_parts: list[dict] = []
        done: set[tuple[str, tuple]] = set()  # (a definition, its referrer's annotations): walked once

        def walk(x: Any, at: str, path: frozenset[str]) -> None:
            nonlocal raw
            x = w.check(x, at)
            if "$ref" in x:
                key, _ = w.resolve(x["$ref"], at)
                rest = {a: b for a, b in x.items() if a != "$ref"}
                seen = (key, _content(rest, self._dumped))
                # a cycle back to a definition being resolved adds no value of its own, and a definition walked
                # already adds the same ones again (a diamond of refs would be walked once per path otherwise); a chain
                # of definitions that are `$ref`s is followed at once (`_Schema.alias_chain`), its last definition
                # walked with the keywords gathered on the way, as following it a definition at a time would
                if key in path or seen in done:
                    return
                done.add(seen)
                view, prefix, keys, last, closes = w.alias_chain(key)
                if closes or not keys.isdisjoint(path):
                    return
                if last != key:
                    seen = (last, _content({**rest, **prefix}, self._dumped))
                    if seen in done:
                        return
                    done.add(seen)
                walk({**rest, **view}, at, path | keys)
                return
            for alt in ("anyOf", "oneOf"):
                if alt in x:
                    rest = {a: b for a, b in x.items() if a not in ("anyOf", "oneOf")}
                    for i, y in enumerate(x[alt]):
                        walk({**rest, **w.check(y, f"{at}/{alt}/{i}")}, f"{at}/{alt}/{i}", path)
                    return
            vals = [x["const"]] if "const" in x else x.get("enum")
            if vals is not None:
                for v in vals:
                    if isinstance(v, str):
                        texts.append(v)
                    else:
                        json_parts.append({"const": v})
                return
            types = _types(x) or (["object"] if "properties" in x else ["array"] if "items" in x else [])
            if not types:
                raw = True  # untyped: the parse keeps any text
                return
            for t in types:
                if t == "string":
                    raw = True
                elif t in ("boolean", "null"):  # the spellings the parse types so (Qwen writes Python's True)
                    texts.extend(_SPELLINGS[t])
                else:
                    json_parts.append({**{a: b for a, b in x.items() if a not in ("type", "nullable")}, "type": t})

        walk(s, where, frozenset())
        body = None if not json_parts else json_parts[0] if len(json_parts) == 1 else {"anyOf": json_parts}
        return raw, list(dict.fromkeys(texts)), body

    def runs(self, w: _Schema, s: dict, params: dict, resolve: Any) -> list[tuple[bytes, Any, bool]]:
        """The parameter blocks' runs (`forced.qwen_call`), from what `params` made of each value; the `%json` items'
        objects closed, as `_json` writes them."""
        props = s.get("properties") or {}
        required = set(s.get("required") or ())
        by_syntax: dict[tuple, forced.Runs] = {}
        out = []
        for k, v in props.items():
            raw, texts, body, _ = self._values[_shape(v)]
            at = f"#/properties/{k}"
            json_run = None
            if body is not None:
                syn = forced.json_syntax(body, closed=True)
                runs = by_syntax.get((syn.key_sep, syn.item_sep))
                if runs is None:
                    runs = by_syntax[(syn.key_sep, syn.item_sep)] = forced.Runs(params, syn, resolve)
                    runs.definitions(params, "#", None)
                json_run = runs.of(body, at)
            head = f"<parameter={k}>\n".encode("utf-8", "surrogatepass")
            out.append((head, forced.qwen_value(raw, texts, json_run, at), k in required))
        return out

    def params(self, w: _Schema, s: dict) -> str:
        props = s.get("properties") or {}
        required = set(s.get("required") or ())
        parts = []
        for k, v in props.items():  # the schema's (the client's) order, as the template renders the tools
            p = self.value_rule(w, k, v, f"#/properties/{k}")
            parts.append(p if k in required else f"{p}?")
        if _open(s) or s.get("additionalProperties", False) is not False:  # more, raw text (the parse types none)
            end = _lit("\n</parameter>\n")
            extra = self.rules.add(f'"<parameter=" /[^>\\n]+/ ">\\n" {_TEXT} {end}', "qextra", "[lazy]")
            parts.append(f"{extra}*")
        return " ".join(parts)


# -- the grammar ----------------------------------------------------------------------------------------------------

def call_grammar(family: str, tools: list[dict], choice: str, name: str | None = None, parallel: bool = True,
                 forcing: bool = True, response_format: str | None = None, triggered: bool = False,
                 ids: dict[str, int] | None = None, max_bytes: int | None = None,
                 thought: str | None = None) -> tuple[str, list[str]]:
    """The grammar text, and the keywords it accepts without enforcing. `tools`: their template form (`api.tools`);
    `choice`: "required" or "named" (with `name`), or "auto" with `response_format` (a `%json` item: the schema's
    branch) or `triggered` (Qwen: free text, then calls). `forcing`: llguidance's canonical forcing (the guidance
    backend; off for auto and xgrammar, as xgrammar's). `ids`: the format's token ids (`output.tool_calls.Family`).
    `max_bytes`: the most grammar text these tools may build (`api.app.App.tool_grammar`'s bound), counted as the rules
    are made, so a schema past it is refused before its grammar is. `thought`: a thinking-on output's thought
    (`grammar.spec.thought`), in front of the calls (and the schema's branch); not with `triggered`, whose free text
    takes the thought."""
    try:
        text, loose = _call_grammar(family, tools, choice, name, parallel, forcing, response_format, triggered,
                                    ids or {}, max_bytes, thought)
    except BoundError as exc:
        raise ToolSchemaError(str(exc)) from None
    except RecursionError as exc:  # a deep enum or const value
        raise ToolSchemaError("the tools' schemas are nested too deeply") from exc
    size = len(text.encode())
    if max_bytes is not None and size > max_bytes:
        raise ToolSchemaError(f"the tools' call grammar is {size} bytes, more than this server builds for their "
                              f"schemas: {max_bytes}")
    if loose:
        log.info("a tool-call grammar accepts %s without enforcing them", ", ".join(loose))
    return text, loose


def _call_grammar(family: str, tools: list[dict], choice: str, name: str | None, parallel: bool, forcing: bool,
                  response_format: str | None, triggered: bool, ids: dict[str, int],
                  max_bytes: int | None, thought: str | None = None) -> tuple[str, list[str]]:
    rules = _Rules(max_bytes)
    caps = Caps("the tools' schemas", "have", family != "gemma4")  # Gemma's grammar writes values as its own text
    calls: list[str] = []
    pending: list[tuple[_Schema, dict | None, Any, _Qwen | None]] = []  # each tool's fixed output, counted once
    for i, tool in enumerate(tools):
        fn = tool["function"]
        if choice == "named" and fn["name"] != name:
            continue
        params = fn.get("parameters")
        caps.prefix = f"tool {fn['name']!r}: "
        w = _Schema(rules, params if isinstance(params, dict) else {}, i, fn["name"], caps)
        root = None
        if isinstance(params, dict) and params:
            json_depth(params, f"tool {fn['name']!r}: its schema")
            w.walk(params, "#")
            root = w.object_root(params)
        q = None
        if family == "gemma4":
            g = _Gemma(rules, f"<[{ids['quote']}]>")
            body = '"{}"' if root is None else g.obj(w, root, "#")
            calls.append(rules.add(f'<[{ids["opener"]}]> {_lit("call:" + fn["name"])} {body} <[{ids["closer"]}]>',
                                   "call"))
        else:
            q = _Qwen(rules)
            ps = q.params(w, root) if root is not None else ""
            calls.append(rules.add(f'<[{ids["opener"]}]> {_lit(chr(10) + "<function=" + fn["name"] + ">" + chr(10))} '
                                   f'{ps} {_lit("</function>" + chr(10))} <[{ids["closer"]}]>'.replace("  ", " "),
                                   "call"))
        pending.append((w, root, params, q))
    if not calls:
        raise ToolSchemaError(f"no tool named {name!r}")
    single = not parallel or choice == "named"
    for w, root, params, q in pending:  # after every tool's walk has passed the caps
        run = _fixed_run(family, w, root, params, q, ids, single)
        if run is None:  # a required parameter without a value (Qwen's `%json` keeps `minItems`)
            raise ToolSchemaError(f"tool {w.name!r}'s call can produce no value")
        forced.cap(run, f"tool {w.name!r}'s call")
    one = rules.add(" | ".join(calls), "calls")
    sep = '"\\n" ' if family == "qwen3_coder" else ""
    many = one if single else f"{one} ({sep}{one})*"
    end = f" <[{ids['end']}]>" if family == "gemma4" else ""
    if triggered:
        think = " | ".join(f"<[{t}]>" for t in ids.get("think", ()))
        free = rules.add(f"(/(.|\\n)+/{' | ' + think if think else ''})*", "free")
        start = f"{free} ({one})?" if not parallel else f"{free} ({one} {free})*"
    else:
        start = f"{many}{end}"
        if response_format is not None:
            start = f"{start} | {response_format}"
        if thought is not None:
            start = f"{thought} ({start})"
    header = "%llguidance {}\n" if forcing else '%llguidance {"no_forcing": true}\n'
    return header + f"start: {start}\n" + rules.text(), sorted(rules.loose)
