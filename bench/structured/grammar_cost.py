#!/usr/bin/env python3
"""What structured output costs the engine's CPU, outside the server: llm-mini-infer's llguidance backend
(`llm_mini_infer.grammar.engine`) on a model's tokenizer.

- **Startup:** building llguidance's tokenizer from tokenizer.json (once per engine).
- **Compile** a new schema: each schema of the corpus (tests/support/schema_corpus.py: the keywords clients use);
  `json_object`. Then the clone a request takes of a compiled grammar (`deep_copy`, of the corpus's first schema).
- **With `--catalog`**, also the catalog schema, a schema with nested `$defs`, nullable enum lists and strict objects
  (tests/data/catalog.schema.json: an object of a count and up to five entries of five nullable enum-list attributes):
  its compile, the first in the process (cold) and again in a fresh compile, and the masks per step: `--rows`
  constrained requests at states along random valid catalog documents, one mask each (a step without drafts) or 1 +
  `--k` along `--k` drafted tokens (a verify step with speculative decoding), filled by llguidance's parallel fill on
  each of `--threads` (the clones of one compiled grammar share its caches).

`--backend guidance --compact` measures guidance's language with `structured_output.disable_any_whitespace` (compact
documents, the tokenizer's eos_token ends a grammar); the default, auto, xgrammar's JSON language through llguidance
(any whitespace, the model's stop ids). The masks are the p50 and p99 of `--reps` repetitions, in ms; the tokenizer
build is one timing (s), each compile one timing (ms; the catalog's repeat compile the least of five), the clone the p50
of 200 (µs).

    uv run python bench/structured/grammar_cost.py                       # the Gemma 4 fixture's tokenizer
    uv run python bench/structured/grammar_cost.py --catalog             # and the catalog schema, with the masks
    uv run python bench/structured/grammar_cost.py --model <model dir> --backend guidance --compact
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.support import paths  # noqa: E402
from tests.support.schema_corpus import CORPUS  # noqa: E402

SCHEMA = json.loads((paths.DATA / "catalog.schema.json").read_text())
MODEL = paths.MODELS_DIR / "gemma-4-31b-it"  # tokenizer and configs (tests/scripts/fetch_fixtures.py)
RESULTS = paths.RESULTS / "structured"


def eos_ids(model: Path) -> list[int]:
    """The model's stop tokens, as the server derives them (llm_mini_infer.config): generation_config.json's
    eos_token_id, else config.json's."""
    f = model / "generation_config.json"
    gen = json.loads(f.read_text()) if f.exists() else {}
    hf = json.loads((model / "config.json").read_text())
    eos = gen.get("eos_token_id", hf.get("text_config", hf).get("eos_token_id", hf.get("eos_token_id")))
    return sorted({int(e) for e in (eos if isinstance(eos, list) else [eos]) if e is not None})


def documents(n: int, rng: random.Random, compact: bool = False) -> list[str]:
    """Valid documents of the catalog schema with random values, as a model would write them (`compact`: without
    whitespace): up to three entries, each list attribute null one time in five, else one or two values (an attribute
    without `minItems` may be empty)."""
    props = SCHEMA["$defs"]["Entry"]["properties"]
    out = []
    for _ in range(n):
        entries = []
        for i in range(rng.randint(0, 3)):
            entry: dict = {"id": i}
            for key, spec in props.items():
                if key == "id":
                    continue
                if "anyOf" not in spec:  # category, year, signed
                    entry[key] = (rng.choice(spec["enum"]) if "enum" in spec else
                                  rng.randint(spec["minimum"], spec["maximum"]) if spec["type"] == "integer" else
                                  rng.random() < 0.5)
                    continue
                arr = spec["anyOf"][0]
                lo = arr.get("minItems", 0)
                entry[key] = None if rng.random() < 0.2 else rng.sample(arr["items"]["enum"], rng.randint(lo, 2))
            entries.append(entry)
        out.append(json.dumps({"total": len(entries), "entries": entries},
                              separators=(",", ":") if compact else None))
    return out


def ms(xs: list[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(xs) * 1e3, q)), 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, default=MODEL, help="a model folder with tokenizer.json and config.json")
    ap.add_argument("--out", type=Path, help="the JSON (default: results/structured/grammar-<time>.json)")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--backend", choices=("auto", "guidance"), default="auto")
    ap.add_argument("--compact", action="store_true", help="disable_any_whitespace (guidance only)")
    ap.add_argument("--catalog", action="store_true", help="also compile the catalog schema and time the masks")
    ap.add_argument("--k", type=int, default=3, help="drafted tokens per row in a verify step")
    ap.add_argument("--rows", default="1,8,32,64", help="constrained requests per step")
    ap.add_argument("--threads", default="1,2,4,8", help="llguidance's fill threads")
    args = ap.parse_args()
    if args.compact and args.backend != "guidance":
        ap.error("--compact needs --backend guidance (as structured_output.disable_any_whitespace)")
    if not (args.model / "tokenizer.json").exists():
        sys.exit(f"{args.model} has no tokenizer.json (uv run tests/scripts/fetch_fixtures.py for the default)")
    from tokenizers import Tokenizer

    from llm_mini_infer import config
    from llm_mini_infer.grammar.engine import LLGuidance
    from llm_mini_infer.grammar.spec import json_object_grammar, json_schema_grammar

    cfg = json.loads((args.model / "config.json").read_text())
    vocab = int(cfg.get("text_config", cfg).get("vocab_size", cfg.get("vocab_size")))
    # what ends a grammar, as the engine sets it: the model's stop ids (auto), the tokenizer's eos_token (guidance)
    eos = (eos_ids(args.model) if args.backend == "auto" else
           config.tokenizer_eos_token(config.load(None, {"model.path": str(args.model)})))
    backend = LLGuidance(args.model / "tokenizer.json", vocab, eos, threads=1)
    res: dict = {"model": args.model.name, "vocab": vocab, "backend": args.backend, "compact": args.compact,
                 "eos": list(backend.eos), "llguidance": backend.version, "k": args.k,
                 "tokenizer_build_s": round(backend.build_s, 3)}
    corpus = {}
    for name, schema, _ in CORPUS:
        t0 = time.perf_counter()
        backend.compile(json_schema_grammar(schema, args.backend, args.compact))
        corpus[name] = round(1e3 * (time.perf_counter() - t0), 3)
    res["compile_corpus_ms"] = corpus
    res["compile_corpus_ms_p50_max"] = [float(np.percentile(list(corpus.values()), 50)), max(corpus.values())]
    t0 = time.perf_counter()
    backend.compile(json_object_grammar(args.backend, args.compact))
    res["compile_json_object_ms"] = round(1e3 * (time.perf_counter() - t0), 3)
    base = backend.compile(json_schema_grammar(CORPUS[0][1], args.backend, args.compact))
    clone = []
    for _ in range(200):
        t0 = time.perf_counter()
        backend.instance(base)
        clone.append(time.perf_counter() - t0)
    res["clone_us_p50"] = round(1e3 * ms(clone, 50), 2)

    if args.catalog:
        catalog = json_schema_grammar(SCHEMA, args.backend, args.compact)
        t0 = time.perf_counter()
        base = backend.compile(catalog)
        res["compile_catalog_cold_ms"] = round(1e3 * (time.perf_counter() - t0), 3)
        times = []
        for _ in range(5):  # the same text again: llguidance's own warm cost (the engine caches compiled grammars)
            t0 = time.perf_counter()
            backend.compile(catalog)
            times.append(time.perf_counter() - t0)
        res["compile_catalog_again_ms"] = [round(1e3 * t, 3) for t in times]

        tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        rng = random.Random(1)
        docs = documents(16, rng, args.compact)
        seqs = [[*tok.encode(d, add_special_tokens=False).ids, backend.eos[0]] for d in docs]
        words = (vocab + 31) // 32
        k = args.k
        masks: dict = {}
        for threads in (int(x) for x in args.threads.split(",")):
            b = LLGuidance(args.model / "tokenizer.json", vocab, eos, threads=threads) if threads > 1 else backend
            c = b.compile(catalog)
            for rows in (int(x) for x in args.rows.split(",")):
                states = []
                for r in range(rows):
                    s = seqs[r % len(seqs)]
                    pos = rng.randrange(0, len(s) - k - 1)
                    m = b.instance(c)
                    assert m.consume_tokens(s[:pos])
                    states.append((m, s[pos:pos + k]))
                buf = np.zeros(((1 + k) * rows, words), np.int32)
                single, drafts = [], []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    b.fill([(m, i) for i, (m, _) in enumerate(states)], [], buf)
                    single.append(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    b.fill([], [(m, (1 + k) * i, d) for i, (m, d) in enumerate(states)], buf)
                    drafts.append(time.perf_counter() - t0)
                masks[f"threads={threads} rows={rows}"] = {"one_mask_ms": [ms(single, 50), ms(single, 99)],
                                                           f"{1 + k}_masks_ms": [ms(drafts, 50), ms(drafts, 99)]}
        res["masks_p50_p99"] = masks
    out = args.out or RESULTS / f"grammar-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(table(res))
    print(f"wrote {out}", file=sys.stderr)


def table(res: dict) -> str:
    """The run as text: the one-off costs, then the masks per step, p50 / p99 in ms, by threads and rows."""
    catalog = (f"; catalog {res['compile_catalog_cold_ms']} ms cold, {min(res['compile_catalog_again_ms'])} ms again"
             if "compile_catalog_cold_ms" in res else "")
    lines = [f"{res['model']}, {res['backend']}{' compact' if res['compact'] else ''}, llguidance {res['llguidance']}",
             f"tokenizer build {res['tokenizer_build_s']} s; compile: corpus p50 "
             f"{res['compile_corpus_ms_p50_max'][0]:.2f} ms, max {res['compile_corpus_ms_p50_max'][1]:.2f} ms; "
             f"json_object {res['compile_json_object_ms']} ms{catalog}; clone {res['clone_us_p50']} us"]
    if "masks_p50_p99" not in res:
        return "\n".join(lines)
    lines += ["", f"{'threads':>7}  {'rows':>4}  {'one mask p50/p99':>16}  {str(1 + res['k']) + ' masks p50/p99':>16}"]
    for key, m in res["masks_p50_p99"].items():
        threads, rows = (part.split("=")[1] for part in key.split())
        one, many = m["one_mask_ms"], m[f"{1 + res['k']}_masks_ms"]
        lines.append(f"{threads:>7}  {rows:>4}  {f'{one[0]:.3f} / {one[1]:.3f}':>16}  "
                     f"{f'{many[0]:.3f} / {many[1]:.3f}':>16}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
