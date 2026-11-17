#!/usr/bin/env python3
"""Validity of structured responses from any OpenAI-compatible server: every response that finishes by stop must
parse and validate against its schema; one cut by max_tokens may not, and must say `length`.

The server gets, through the chat completions API, non-streamed and streamed, at `--conc` requests at a time:
- **corpus:** every schema of tests/support/schema_corpus.py (the keywords clients use: bounds, patterns, formats,
  optional keys, `anyOf`/`oneOf`, recursion, tuples, dictionaries, top-level arrays and strings) with its prompt, greedy
  and at temperature 0.8 (two seeds);
- **catalog** (`--n-catalog`, default none): that many requests to catalogue the items in a box, each with the catalog
  schema, a schema with nested `$defs`, nullable enum lists and strict objects (tests/data/catalog.schema.json: up to
  five entries of five nullable enum-list attributes), as a `json_schema` response format, two in three strict; greedy
  and at temperature 0.8 in turn.
Each response that ends by stop is parsed (`json.loads`) and validated (`jsonschema`, Draft 2020-12, with the format
checks it has without optional packages: no `time`, `date-time` or `uri`). One that repeats a key in an object is
counted apart: JSON parsers keep the last value, so a schema's counts (`minProperties`) can fail on a document the
grammar accepted, and no grammar can exclude repeated free-form keys (`additionalProperties` maps). A response cut by
max_tokens must report `length`.

    uv run llm-mini-infer serve --model <model> --port 8000 &       # or vLLM, SGLang, ...
    uv run python bench/structured/validity.py --base-url http://127.0.0.1:8000 --model <served name>

The JSON report (`--out`) holds the summary and each response's text, finish reason and verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.support import paths  # noqa: E402
from tests.support.catalog_schema import PATH  # noqa: E402
from tests.support.catalog_schema import prompt as catalog_prompt  # noqa: E402
from tests.support.schema_corpus import CORPUS  # noqa: E402

SCHEMA = json.loads(PATH.read_text())
RESULTS = paths.RESULTS / "structured"
DEFAULT_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


def response_format(schema: dict, name: str, strict: bool = True) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": strict, "schema": schema}}


def build_requests(n_catalog: int, max_tokens: int) -> list[dict[str, Any]]:
    out = []
    for i in range(n_catalog):
        out.append({"group": "catalog", "id": f"catalog-{i}", "schema": SCHEMA, "stream": i % 4 >= 2, "body": {
            "messages": [{"role": "user", "content": catalog_prompt(i)}], "max_tokens": max_tokens,
            "temperature": 0.0 if i % 2 == 0 else 0.8, "seed": i,
            "response_format": response_format(SCHEMA, "catalog", strict=i % 3 != 0)}})
    for name, schema, prompt in CORPUS:
        for j, (temp, seed) in enumerate(((0.0, 0), (0.8, 1), (0.8, 2))):
            out.append({"group": "corpus", "id": f"{name}-{j}", "schema": schema, "stream": j == 1, "body": {
                "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temp,
                "seed": seed, "response_format": response_format(schema, name)}})
    return out


async def send(session: Any, url: str, headers: dict, model: str, extra: dict, req: dict[str, Any]) -> dict:
    body = {"model": model, **extra, **req["body"], "stream": req["stream"]}
    if req["stream"]:
        body["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    async with session.post(url, json=body, headers=headers) as resp:
        if resp.status != 200:
            return {"error": f"HTTP {resp.status}: {(await resp.text())[:300]}"}
        if not req["stream"]:
            r = await resp.json()
            ch = r["choices"][0]
            return {"text": ch["message"]["content"], "finish": ch["finish_reason"],
                    "tokens": r["usage"]["completion_tokens"], "s": time.perf_counter() - t0}
        text, finish, tokens = [], None, None
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            obj = json.loads(line[5:])
            if obj.get("usage"):
                tokens = obj["usage"]["completion_tokens"]
            for ch in obj.get("choices") or []:
                text.append((ch.get("delta") or {}).get("content") or "")
                finish = ch.get("finish_reason") or finish
        return {"text": "".join(text), "finish": finish, "tokens": tokens, "s": time.perf_counter() - t0}


class DuplicateKeys(ValueError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(set(keys)) < len(keys):
        raise DuplicateKeys(sorted({k for k in keys if keys.count(k) > 1}))
    return dict(pairs)


def validate(text: str, schema: dict) -> str | None:
    """None if `text` is a JSON document valid against `schema` (Draft 2020-12 and jsonschema's format checks: the
    module docstring says which); else what is wrong, a repeated key reported apart (`duplicate keys`)."""
    import jsonschema

    try:
        doc = json.loads(text, object_pairs_hook=_pairs)
    except DuplicateKeys as exc:
        return f"duplicate keys: {exc.args[0]}"
    except ValueError as exc:
        return f"not JSON: {exc}"
    try:
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
                                        ).validate(doc)
    except jsonschema.ValidationError as exc:
        return f"invalid: {exc.message[:200]}"
    return None


async def run_all(args: argparse.Namespace, reqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import aiohttp

    key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    headers = {"authorization": f"Bearer {key}"} if key else {}
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(args.conc)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        async def one(req: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                try:
                    res = await send(session, url, headers, args.model, args.extra_body, req)
                except Exception as exc:  # a connection error or a timeout
                    res = {"error": type(exc).__name__}
            res.update(id=req["id"], group=req["group"], stream=req["stream"],
                       temperature=req["body"]["temperature"])
            if res.get("finish") == "stop":
                res["error_schema"] = validate(res["text"], req["schema"])
            return res

        return list(await asyncio.gather(*(one(r) for r in reqs)))


def summarize(results: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in ("catalog", "corpus", "all"):
        rs = [r for r in results if group == "all" or r["group"] == group]
        stop = [r for r in rs if r.get("finish") == "stop"]
        length = [r for r in rs if r.get("finish") == "length"]
        bad = [r for r in stop if r.get("error_schema") and not r["error_schema"].startswith("duplicate keys")]
        dup = [r for r in stop if (r.get("error_schema") or "").startswith("duplicate keys")]
        errors = sum("error" in r for r in rs)
        out[group] = {"requests": len(rs), "stop": len(stop), "valid": len(stop) - len(bad) - len(dup),
                      "invalid": len(bad), "duplicate_keys": len(dup),
                      "duplicate_key_ids": [(r["id"], r["error_schema"]) for r in dup][:20],
                      "length": len(length), "length_at_max_tokens": sum(r["tokens"] == max_tokens for r in length),
                      "errors": errors, "other_finish": len(rs) - len(stop) - len(length) - errors,
                      "invalid_ids": [(r["id"], r["error_schema"]) for r in bad][:20],
                      "mean_tokens": sum(r.get("tokens") or 0 for r in rs) / max(1, len(rs))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="the server (default: %(default)s)")
    ap.add_argument("--model", required=True, help="the served model's name")
    ap.add_argument("--out", type=Path, help="the JSON report (default: results/structured/validity-<time>.json)")
    ap.add_argument("--n-catalog", type=int, default=0, help="also this many catalog-schema requests (default: none)")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--extra-body", type=json.loads, default=DEFAULT_EXTRA,
                    help=f"JSON merged into every request (default: {json.dumps(DEFAULT_EXTRA)}; a value replaces it)")
    ap.add_argument("--api-key", help="a bearer token (default: OPENAI_API_KEY, if set)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per request, seconds")
    args = ap.parse_args()
    reqs = build_requests(args.n_catalog, args.max_tokens)
    t0 = time.time()
    results = asyncio.run(run_all(args, reqs))
    wall = time.time() - t0
    summary = summarize(results, args.max_tokens)
    out = args.out or RESULTS / f"validity-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "wall_s": wall, "max_tokens": args.max_tokens,
                               "results": results}, indent=1) + "\n")
    a = summary["all"]
    for group in ("catalog", "corpus"):
        g = summary[group]
        if not g["requests"]:
            continue
        print(f"{group:>6}: {g['valid']}/{g['stop']} stop responses valid, {g['duplicate_keys']} repeat a key, "
              f"{g['invalid']} invalid; {g['length']} cut by max_tokens; {g['errors']} errors; "
              f"{g['mean_tokens']:.0f} tokens on average")
    problems = a["invalid"] + a["errors"] + a["other_finish"]
    print(f"{a['valid']}/{a['stop']} stop responses valid ({a['duplicate_keys']} more repeat a key); {a['length']} cut "
          f"by max_tokens ({a['length_at_max_tokens']} at exactly max_tokens); {problems} problems; {wall:.0f} s",
          flush=True)
    print(f"wrote {out}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
