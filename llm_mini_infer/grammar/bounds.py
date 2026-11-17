"""The JSON Schema subset a grammar is built from, and its caps (architecture.md, "Structured output"; operations.md,
"Settings", gives the numbers).

llguidance's compile cost follows a schema's shape: keywords merged with each other (an `anyOf` beside `properties`, a
`$ref` beside an `anyOf`) multiply its work level by level, and `patternProperties` grows as a power of its keys
(guidance-ai/llguidance#436 and #438). So a schema is checked first, in one linear walk of the document as sent, as
OpenAI's strict Structured Outputs checks one:
- **keywords** (a `response_format`; the tools' list is `grammar.tools.KNOWN`): `REFUSED`, the keywords llguidance acts
  on that the subset leaves out, are a 400; any other key the subset does not name is an annotation, walked past, as
  llguidance ignores it. An `$id` stands at the root only, and so does draft-04's `id` (not a fragment) in a document
  that names draft-04 (llguidance resolves the `$ref`s under a nested one against it: llguidance#444); a `contentSchema`
  holds neither (its resolver reads one there, and it can stand for the root: llguidance#442);
- **no merging** (`keep_apart`): `anyOf`, `oneOf`, `$ref` and a one-member `allOf` (read as that member) stand beside
  `ANNOTATIONS`, keys starting `x-` and definitions alone (llguidance merges any other key into them);
- **the caps:** `MAX_NESTING` levels, counting the definitions `$ref`s reach where they are used (`deepest`); the
  counts `Caps` keeps, a definition counted once however many `$ref`s name it; `JSON_DEPTH` (`json_depth`). A
  `response_format` is checked against vLLM's `MAX_JSON_NESTING` first (`json_nesting`, with vLLM's text), so
  `JSON_DEPTH` bounds the tools' schemas, which vLLM's cap does not reach here.

At the caps, the costliest shapes measured compile in under 0.3 s and 500 MB, in the frontend's check and the engine's
compile alike.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any
from urllib.parse import unquote

from . import NestingError

MAX_NESTING = 32
MAX_OBJECT_PROPERTIES = 400  # in one object (tuple items in one array), below llguidance's 2,000 items a parser row
MAX_PROPERTIES = 5_000  # in all, tuple items included
MAX_VALUES = 1_000  # enum and const values, in all
MAX_VALUE_ITEMS = 10_000  # the array items and object members inside them, in all: their compile grows faster
MAX_ALTERNATIVES = 1_000  # anyOf and oneOf members, in all
MAX_DEFINITIONS = 1_000
MAX_PATTERN_BYTES = 16_384  # `pattern` and `x-guidance` strings, in all: 16 KB of alternations compile in 0.04 s
MAX_FORMATS = 400  # strings with a format llguidance checks, in all: 400 `uri`s compile in 0.13 s, 1,000 in 0.76 s
# An object under `minProperties` or `maxProperties` with optional properties: its grammar grows with the square of
# their count (`Caps.counted`), so both are capped
MAX_COUNTED = 32  # optional properties in one object under `minProperties`
MAX_COUNTED_MEMBERS = 40_000  # what all such objects count, in all: 3 to 4 ms per 1,000 in the check
FORMATS = frozenset(("date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid", "uri"))
I64 = (-(1 << 63), (1 << 63) - 1)  # llguidance's integers
LOWEST = -2.0 ** 63  # the lowest 64-bit integer, as a double: integer bounds at or near it are outside the range
PATHS = 100_000  # the steps `deepest` takes to find a group's deepest path before it bounds it by the group's size

ANNOTATIONS = frozenset(("title", "description", "default", "examples", "$comment", "deprecated", "readOnly",
                         "writeOnly", "$schema", "$id", "$anchor", "discriminator", "id", "contentMediaType",
                         "contentEncoding"))
FACETS = frozenset(("pattern", "format", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
                    "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "minProperties", "maxProperties",
                    "uniqueItems"))
DEFS = ("$defs", "definitions")
KEYWORDS = ANNOTATIONS | FACETS | frozenset(("type", "nullable", "enum", "const", "properties", "required",
                                             "additionalProperties", "items", "prefixItems", "$ref", "anyOf", "oneOf",
                                             "allOf", *DEFS))
MERGING = ("$ref", "anyOf", "oneOf", "allOf")
# The keywords of JSON Schema drafts 3 to 2020-12 and OpenAPI's that llguidance 1.9.1 acts on and the subset leaves
# out: each changes its verdict or its grammar, beside a type or alone, under some draft's `$schema`. Every other
# name the subset leaves out changes neither (`tests/structured/test_keywords.py` holds the enumeration and checks both)
REFUSED = frozenset(("not", "if", "then", "else", "dependencies", "dependentRequired", "dependentSchemas", "contains",
                     "minContains", "maxContains", "propertyNames", "patternProperties", "additionalItems",
                     "unevaluatedItems", "unevaluatedProperties", "$dynamicRef", "$dynamicAnchor", "$recursiveRef",
                     "$recursiveAnchor"))
JSON_DEPTH = 124  # the deepest document llguidance parses under every backend (serde_json's 128, less guidance's wrap)
MAX_JSON_NESTING = 64  # a `response_format` document's JSON levels: vLLM's MAX_STRUCTURED_OUTPUT_JSON_NESTING


class BoundError(ValueError):
    """A schema outside the subset or past a cap (a 400)."""


def annotation(k: str) -> bool:
    return k in ANNOTATIONS or k.startswith("x-")


def keep_apart(x: dict, at: str, inert: Callable[[str], bool] = annotation) -> None:
    """The merging rule: `anyOf`, `oneOf`, `$ref` or `allOf` with annotations (`inert`) and definitions alone beside it,
    and an `allOf` of one member."""
    own = next((k for k in MERGING if k in x), None)
    if own is None:
        return
    for k in x:
        if k != own and k not in DEFS and not inert(k):
            raise BoundError(f"{k!r} beside {own!r} at {at} is not supported: {own!r} takes only annotations beside it")
    if own == "allOf" and (not isinstance(x["allOf"], list) or len(x["allOf"]) != 1):
        n = len(x["allOf"]) if isinstance(x["allOf"], list) else 0
        raise BoundError(f"'allOf' at {at} has {n} members: only one is supported")


class Caps:
    """The counts the caps bound, over the schemas of one grammar: `what` names them in a message, `prefix` leads the
    messages about one schema. `json`: llguidance compiles the schemas' values, bounds and formats as JSON Schema (a
    `response_format`, Qwen's `%json` items), not as the grammar's own text (Gemma's)."""

    def __init__(self, what: str = "the JSON schema", has: str = "has", json: bool = True) -> None:
        self.what, self.has, self.prefix, self.json = what, has, "", json
        self.properties = self.values = self.items = self.alternatives = self.definitions = 0
        self.pattern_bytes = self.formats = self.counted_members = 0

    def add(self, x: dict, at: str) -> None:
        """What schema `x`, at JSON pointer `at`, holds: an object's properties (with the names it requires), an
        array's tuple items, its enum and const values (and their items and members), its alternatives, its pattern
        and `x-guidance` text, and a format llguidance checks; and integers outside the supported range
        (`integer_range`)."""
        props, req = x.get("properties"), x.get("required")
        names = set(props) if isinstance(props, dict) else set()
        if isinstance(req, list):
            names.update(r for r in req if isinstance(r, str))
        self._one(len(names), "an object", "properties", at)
        if isinstance(props, dict):
            required = {r for r in req if isinstance(r, str)} if isinstance(req, list) else set()
            self.counted(x, sum(k not in required for k in props), len(required), at)
        for k in ("prefixItems", "items"):
            if isinstance(x.get(k), list):
                self._one(len(x[k]), "an array", "tuple items", at)
        enum = x.get("enum")
        values = [*(enum if isinstance(enum, list) else ()), *((x["const"],) if "const" in x else ())]
        n_values = len(values)
        self.values += len(values)
        self._all(self.values, "enum and const values", MAX_VALUES)
        if self.json:
            integer_range(x, values, at)
        while values:  # their items and members, iteratively (`json_depth` bounds them already)
            v = values.pop()
            if isinstance(v, dict | list):
                self.items += len(v)
                self._all(self.items, "array items and object members in enum and const values", MAX_VALUE_ITEMS)
                values += v.values() if isinstance(v, dict) else v
        self.alternatives += sum(len(x[k]) for k in ("anyOf", "oneOf") if isinstance(x.get(k), list))
        self._all(self.alternatives, "'anyOf' and 'oneOf' alternatives", MAX_ALTERNATIVES)
        own = x.get("x-guidance")
        texts = [x.get("pattern"), *(own.get(k) for k in ("whitespace_pattern", "item_separator", "key_separator")
                                     if isinstance(own, dict))]
        self.pattern_bytes += sum(len(t.encode("utf-8", "surrogatepass")) for t in texts if isinstance(t, str))
        self._all(self.pattern_bytes, "bytes of 'pattern' and 'x-guidance' text", MAX_PATTERN_BYTES)
        types = _types(x)
        fmt = x.get("format")
        if self.json and isinstance(fmt, str) and fmt in FORMATS and (not types or "string" in types):
            self.formats += max(1, n_values)  # beside values, llguidance intersects it with each
            self._all(self.formats, "strings with a 'format'", MAX_FORMATS)

    def counted(self, x: dict, optional: int, required: int, at: str) -> None:
        """An object under `minProperties` or `maxProperties` with `optional` properties: llguidance writes it once for
        each, with every member (one of them required: `minProperties` one more than the required ones) or with the
        required ones and that one (`maxProperties` one more), when no other key is allowed (`additionalProperties`
        false, or any schema without a value); with another count, or other keys allowed, it refuses the object
        itself."""
        lo, hi = x.get("minProperties"), x.get("maxProperties")
        lo = lo - required if _count(lo) and lo > required else 0
        hi = max(hi - required, 0) if _count(hi) else None
        if not optional or (lo == 0 and hi is None) or lo > 1 or (hi is not None and hi != 1):
            return  # (another count, or other keys allowed: llguidance refuses the object itself)
        if hi is None and optional > MAX_COUNTED:
            raise BoundError(f"{self.prefix}an object at {at} has {optional:,} optional properties under "
                             f"'minProperties', more than {MAX_COUNTED}")
        self.counted_members += (optional + 1) * (required + 1) if hi == 1 else optional * (required + optional)
        self._all(self.counted_members, "members in the copies of its objects under 'minProperties' or "
                  "'maxProperties'", MAX_COUNTED_MEMBERS)

    def defined(self, n: int) -> None:
        self.definitions += n
        if self.definitions > MAX_DEFINITIONS:
            raise BoundError(f"{self.what} {self.has} {self.definitions:,} definitions, more than {MAX_DEFINITIONS:,}")

    def _one(self, n: int, kind: str, name: str, at: str) -> None:
        if n > MAX_OBJECT_PROPERTIES:
            raise BoundError(f"{self.prefix}{kind} at {at} has {n:,} {name}, more than {MAX_OBJECT_PROPERTIES}")
        self.properties += n
        self._all(self.properties, "properties and tuple items", MAX_PROPERTIES)

    def _all(self, n: int, name: str, cap: int) -> None:
        if n > cap:
            raise BoundError(f"{self.what} {self.has} {n:,} {name} in all, more than {cap:,}")


def _count(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _types(x: dict) -> list:
    t = x.get("type")
    return [t] if isinstance(t, str) else list(t) if isinstance(t, list) else []


def _sinks(v: Any) -> bool:
    """Whether a JSON number reads as a double at or below the lowest 64-bit integer (`LOWEST`): an integer at most
    -2**63 + 512, a float at most the next double up."""
    if isinstance(v, bool) or not isinstance(v, int | float):
        return False
    if isinstance(v, float):
        return v <= LOWEST + 1024
    try:
        return float(v) <= LOWEST
    except OverflowError:  # an integer past a double's range
        return v < 0


def integer_range(x: dict, values: list, at: str) -> None:
    """A `BoundError` for an integer outside the supported range in schema `x`: an upper bound of an integer type at
    the bottom of the 64-bit range (`_sinks`), and such a number in an `enum` or `const` value that llguidance may
    compile as an integer: one within 64 bits, or any beside an integer type or a structure that may hold one. A lower
    bound alone, a `number` and a float alone are in range."""
    integer = "integer" in _types(x)
    if integer:
        for k in ("maximum", "exclusiveMaximum"):
            if _sinks(x.get(k)):
                raise BoundError(f"{k!r} {x[k]} at {at} is not supported: an integer upper bound outside the "
                                 "supported integer range")
    if not values:
        return
    beside = integer or any(k in x for k in ("items", "prefixItems", "properties", "additionalProperties"))
    todo = list(values)
    while todo:
        v = todo.pop()
        if isinstance(v, dict | list):
            todo += v.values() if isinstance(v, dict) else v
        elif _sinks(v) and (beside or (isinstance(v, int) and I64[0] <= v <= I64[1])):
            raise BoundError(f"the number {v} in a value at {at} is not supported: an integer outside the supported "
                             "integer range")


def components(root: Any, succ: Callable[[Any], Iterable[Any]]) -> Iterator[list[Any]]:
    """Tarjan's strongly connected components of the graph from `root`, without recursion: each component's members
    (in the order they leave the path) as it closes, successors first; `succ` gives a node's successors, asked once."""
    index, low = {root: 0}, {root: 0}
    on, path = {root}, [root]
    work = [(root, iter(succ(root)))]
    while work:
        v, it = work[-1]
        for w in it:
            if w not in index:
                index[w] = low[w] = len(index)
                path.append(w)
                on.add(w)
                work.append((w, iter(succ(w))))
                break
            if w in on:
                low[v] = min(low[v], index[w])
        else:
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                members = []
                while True:
                    w = path.pop()
                    on.discard(w)
                    members.append(w)
                    if w == v:
                        break
                yield members


def deepest(height: dict[Any, int], uses: dict[Any, dict[Any, int]]) -> int:
    """The deepest a schema nests, from None (its root): a path enters a definition at its root, goes down to a `$ref`
    (that `$ref`'s level) and enters the definition it names, and ends inside one (its depth, `height`); no definition
    twice, as llguidance compiles each once. `uses`: the definitions each one's `$ref`s name, at their deepest level.
    Through the DAG of strongly connected components (Tarjan's algorithm, successors first) each step is exact; inside
    a component, its deepest path is searched for, and past `PATHS` steps bounded by the sum of its members' `$ref`
    levels."""
    comp: dict[Any, int] = {}
    best: dict[Any, int] = {}
    for n_comp, members in enumerate(components(None, lambda v: sorted(uses.get(v, {}), key=str))):
        for w in members:
            comp[w] = n_comp
        _component(members, n_comp, comp, height, uses, best)
    return best[None]


def _component(members: list, c: int, comp: dict, height: dict, uses: dict, best: dict) -> None:
    """`best` for each member of component `c`: the deepest path from its root (its successors' are known)."""
    out = {}  # ending in a member, or leaving the component from it
    for m in members:
        named = uses.get(m, {})
        out[m] = max([height[m], *(lv + best[w] for w, lv in named.items() if comp[w] != c)])
    inner = {m: [(w, lv) for w, lv in uses.get(m, {}).items() if comp[w] == c] for m in members}
    if not any(inner.values()):  # one definition, no $ref to itself
        best.update(out)
        return
    steps = 0
    for start in members:
        top, seen = out[start], {start}
        stack = [(start, 0, iter(inner[start]))]
        while stack and steps <= PATHS:
            m, acc, it = stack[-1]
            for w, lv in it:
                steps += 1
                if w not in seen:
                    seen.add(w)
                    top = max(top, acc + lv + out[w])
                    stack.append((w, acc + lv, iter(inner[w])))
                    break
            else:
                stack.pop()
                seen.discard(m)
        if steps > PATHS:  # too many paths to search: the sum of every member's deepest $ref within it
            bound = sum(max(lv for _, lv in inner[m]) for m in members if inner[m]) + max(out.values())
            best.update(dict.fromkeys(members, bound))
            return
        best[start] = top


def _key(name: str) -> str:
    """A key as a JSON pointer writes it."""
    return name.replace("~", "~0").replace("/", "~1")


def ref_name(ref: Any, at: str) -> tuple[str, str] | None:
    """A `$ref` as llguidance reads it: `#` (None), or the definitions' key and the name in `#/$defs/NAME` or
    `#/definitions/NAME`, the whole pointer percent-decoded before it is split (a name that holds a `/` is a pointer
    into a definition). Any other `$ref` is a `BoundError`."""
    if ref == "#":
        return None
    if isinstance(ref, str) and ref.startswith("#/"):
        parts = unquote(ref[2:]).split("/")
        if len(parts) == 2 and parts[0] in DEFS:
            return parts[0], parts[1].replace("~1", "/").replace("~0", "~")
    raise BoundError(f"the $ref {ref!r} at {at} is not supported: a $ref names '#', '#/$defs/NAME' or "
                     "'#/definitions/NAME', and NAME holds no '/'")


def def_key(ref: Any, defs: dict) -> str | None:
    """The definition a tool's `$ref` names among `defs` (its `$defs` and `definitions`, merged), read as llguidance
    reads it (`ref_name`), behind a literal `#/$defs/` or `#/definitions/` (Qwen's items rewrite those prefixes alone);
    None when it does not resolve. The grammars and the typing of the arguments both resolve names so."""
    if not isinstance(ref, str) or not ref.startswith(("#/$defs/", "#/definitions/")):
        return None
    try:
        got = ref_name(ref, "")
    except BoundError:
        return None
    return got[1] if got is not None and got[1] in defs else None


def definition(root: dict, ref: Any, at: str) -> Any:
    """The node a `$ref` names: the root (`#`) or one of its definitions (`ref_name`); None when the definition is
    absent (llguidance's own 400)."""
    got = ref_name(ref, at)
    if got is None:
        return root
    defs = root.get(got[0])
    return defs.get(got[1]) if isinstance(defs, dict) else None


def _identified(x: Any) -> bool:
    """Whether an `$id` or an `id` stands anywhere in `x`."""
    todo = [x]
    while todo:
        y = todo.pop()
        if isinstance(y, dict):
            if "$id" in y or "id" in y:
                return True
            todo += y.values()
        elif isinstance(y, list):
            todo += y
    return False


def json_depth(root: Any, what: str) -> None:
    """A `BoundError` when the document nests deeper than `JSON_DEPTH` JSON levels (the root is the first)."""
    todo = [(root, 1)]
    while todo:
        x, depth = todo.pop()
        if depth > JSON_DEPTH:
            raise BoundError(f"{what} nests more than {JSON_DEPTH} JSON levels deep, the most llguidance parses")
        todo += [(v, depth + 1) for v in (x.values() if isinstance(x, dict) else x) if isinstance(v, dict | list)]


def json_nesting(root: dict | list) -> None:
    """vLLM's cap on a `response_format` schema: a `NestingError` with its text when the document's objects and arrays
    nest more than `MAX_JSON_NESTING` levels, the root the first (vLLM's `check_json_nesting`, walked level by level as
    it walks)."""
    level = [root]
    for _ in range(MAX_JSON_NESTING):
        level = [v for x in level for v in (x.values() if isinstance(x, dict) else x) if isinstance(v, dict | list)]
        if not level:
            return
    raise NestingError("The JSON schema is nested too deeply: objects and arrays may nest at most "
                       f"{MAX_JSON_NESTING} levels")


def check_json_schema(root: Any, what: str = "the JSON schema") -> None:
    """A `BoundError` unless `root` keeps the subset and its caps (module docstring). Each schema the document holds
    is walked once, inside the definition that holds it."""
    if not isinstance(root, dict):
        return
    json_depth(root, what)
    caps = Caps(what)
    nested_id: str | None = None  # where a nested draft-04 `id` is, which a document naming draft-04 refuses
    draft4 = False
    height: dict[Any, int] = {}
    uses: dict[Any, dict[Any, int]] = {}
    starts: list[tuple[Any, str, Any]] = [(root, "#", None)]  # the root, then each definition
    while starts:
        start, where, owner = starts.pop()
        height[owner] = 0
        named = uses.setdefault(owner, {})
        stack: list[tuple[Any, str, int]] = [(start, where, 1)]
        while stack:
            x, at, level = stack.pop()
            if not isinstance(x, dict):
                continue  # a boolean schema, or a value llguidance refuses
            height[owner] = max(height[owner], level)
            for k in x:
                if k in REFUSED:
                    raise BoundError(f"the keyword {k!r} at {at} is not supported")
            if "$id" in x and x is not root:
                raise BoundError(f"the '$id' at {at} is not supported: only the root's is")
            if "contentSchema" in x and _identified(x["contentSchema"]):  # walked past, but read by its resolver
                raise BoundError(f"the 'contentSchema' at {at} is not supported: llguidance reads an '$id' inside it, "
                                 "which can stand for the root")
            if isinstance(x.get("id"), str) and not x["id"].startswith("#") and x is not root and nested_id is None:
                nested_id = at
            draft4 |= isinstance(x.get("$schema"), str) and "draft-04" in x["$schema"]
            keep_apart(x, at)
            caps.add(x, at)
            if "$ref" in x:
                t = definition(root, x["$ref"], at)
                if isinstance(t, dict):
                    w = None if t is root else id(t)
                    named[w] = max(named.get(w, 0), level)
            for k in DEFS:
                if isinstance(x.get(k), dict):
                    caps.defined(len(x[k]))
                    starts += [(d, f"{at}/{k}/{_key(n)}", id(d)) for n, d in x[k].items()]
            props = x.get("properties")
            if isinstance(props, dict):
                stack += [(v, f"{at}/properties/{_key(n)}", level + 1) for n, v in props.items()]
            for k in ("items", "additionalProperties"):
                if isinstance(x.get(k), dict):
                    stack.append((x[k], f"{at}/{k}", level + 1))
            for k in ("anyOf", "oneOf", "prefixItems", "items"):
                if isinstance(x.get(k), list):
                    stack += [(v, f"{at}/{k}/{i}", level + 1) for i, v in enumerate(x[k])]
            if isinstance(x.get("allOf"), list):
                stack.append((x["allOf"][0], f"{at}/allOf/0", level))  # read as its member
    if draft4 and nested_id is not None:
        raise BoundError(f"the 'id' at {nested_id} is not supported under draft-04: only the root's is")
    depth = deepest(height, uses)
    if depth > MAX_NESTING:
        raise BoundError(f"{what} nests {depth} levels deep, counting its $refs' definitions where they are used, "
                         f"more than {MAX_NESTING}")
