"""Structured output under `structured_output.backend: guidance` is vLLM's under `--structured-outputs-config
{"backend": "guidance", "disable_any_whitespace": true}`, proven against vLLM's own code (tests/support/vllm_ref.py;
skipped without a vLLM checkout):

(A) a request's grammar text: the frontend's (`grammar.spec`) is vLLM's `serialize_guidance_grammar`, on the catalog
    schema, the corpus and `json_object`, compact and with any whitespace;
(B) the engine's LLTokenizer (built from tokenizer.json, the tokenizer's `eos_token` its EOS) is vLLM's
    (`GuidanceBackend`: `llguidance.hf.from_tokenizer` over transformers' tokenizer) on every Gemma 4 id;
(C) the masks: along random documents, every mask the engine fills, one per row or 1 + 3 along a row's drafts, is what
    vLLM's StructuredOutputManager fills (the drafts first cut to the grammar's prefix by `validate_tokens`, as its
    scheduler does), wherever verification reads it, on the catalog schema. The rows past a drafted EOS stay skipped:
    under ignore_eos this server verifies the drafts past it, where vLLM cuts them (a documented deviation).

vLLM's code runs on the installed llguidance (1.9.1). The release that `REV` names (tests/support/vllm_ref.py) pins
it below 1.8 (`>= 1.7.0, < 1.8.0`), whose masks differ from 1.9.1's after the key of a `multipleOf` number and in the
keys an object takes beside its named properties (grammar/spec.py): none of these schemas has either, so (C) holds
against both.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from llm_mini_infer import config
from llm_mini_infer.api.protocol import decode_chat
from llm_mini_infer.grammar.engine import words
from llm_mini_infer.grammar.spec import GrammarSpecs, json_object_grammar, json_schema_grammar
from llm_mini_infer.runner.fake import unpack_mask

from ..support import schema_corpus, vllm_ref
from ..support.catalog_schema import NAME as CATALOG
from ..support.catalog_schema import PATH as CATALOG_SCHEMA
from ..support.paths import MODELS_DIR

GEMMA = MODELS_DIR / "gemma-4-31b-it"
SYNTHETIC = CATALOG_SCHEMA  # tests/support/catalog_schema.py
VOCAB = 262_144
EOS = 1  # Gemma's <eos>: the tokenizer's eos_token
# The first masks on both schemas: '{"' (forced), the first key's token (forced), then '":' or '"' (or its byte
# token): Gemma has ':-' tokens and the first property is an integer without a minimum, so llguidance cannot force
# '":' (with "minimum": 0 the third mask is {1083})
FIRST = ({14937}, {6725}, {272, 1083, 236775})
SPECIAL = (50, 106)  # Gemma's other stop ids (generation_config.json): never allowed under a guidance mask

pytestmark = pytest.mark.skipif(vllm_ref.src() is None, reason="no vLLM checkout (LLM_MINI_INFER_VLLM_SRC, default "
                                                               "../vllm)")



def params(**kw: Any) -> SimpleNamespace:
    """vLLM's StructuredOutputsParams, the fields `get_structured_output_key` reads."""
    return SimpleNamespace(**{"json": None, "json_object": None, "regex": None, "choice": None, "grammar": None,
                              "structural_tag": None, **kw})


def chat_body(schema: dict, name: str = CATALOG) -> bytes:
    return json.dumps({"messages": [{"role": "user", "content": "Describe the catalogue."}],
                       "response_format": {"type": "json_schema",
                                           "json_schema": {"name": name, "schema": schema, "strict": True}}}).encode()


@pytest.fixture(scope="module")
def offline():
    """No Hugging Face request and no token: the tokenizers are local (transformers checks the Hub for some)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.delenv("HF_TOKEN", raising=False)
        mp.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        mp.setenv("LLGUIDANCE_LOG_LEVEL", "0")  # quiet: llguidance's warnings quote the schema, which may be private
        yield


@pytest.fixture(scope="module")
def text_ns() -> dict[str, Any]:
    return vllm_ref.load_text(vllm_ref.src())


@pytest.fixture(scope="module")
def grammar_ns(offline) -> dict[str, Any]:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return vllm_ref.load_grammar(vllm_ref.src())


@pytest.fixture(scope="module")
def engine(offline):
    """The engine's grammars under guidance with compact JSON (`engine.process._grammars`), and the config."""
    if not (GEMMA / "tokenizer.json").exists():
        pytest.skip("tests/.fixtures/models/gemma-4-31b-it is absent: uv run tests/scripts/fetch_fixtures.py")
    from llm_mini_infer.engine.process import _grammars

    cfg = config.derive(config.load(None, {"model.path": str(GEMMA), "runner.kind": "fake",
                                           "structured_output.backend": "guidance",
                                           "structured_output.disable_any_whitespace": True}))
    grammars = _grammars(cfg, SimpleNamespace(grammar_masks=True, vocab_size=cfg.vocab_size, backend="fake"))
    return cfg, grammars


@pytest.fixture(scope="module")
def hf(offline):
    transformers = pytest.importorskip("transformers")
    if not (GEMMA / "tokenizer.json").exists():
        pytest.skip("tests/.fixtures/models/gemma-4-31b-it is absent: uv run tests/scripts/fetch_fixtures.py")
    return transformers.AutoTokenizer.from_pretrained(GEMMA, local_files_only=True)


# -- (A) the grammar text ----------------------------------------------------------------------------------------------


def corpus() -> list[tuple[str, dict]]:
    return [(CATALOG, json.loads(SYNTHETIC.read_text())), *((n, s) for n, s, _ in schema_corpus.CORPUS)]


@pytest.mark.parametrize("compact", [True, False], ids=["compact", "any_whitespace"])
def test_guidance_text_is_vllms(text_ns, compact) -> None:
    """Ours from the frontend's decode of a request (msgspec: the schema's text as received), vLLM's from its
    json.dumps of the parsed schema."""
    key, serialize = text_ns["get_structured_output_key"], text_ns["serialize_guidance_grammar"]
    specs = GrammarSpecs(backend="guidance", disable_any_whitespace=compact)
    schemas = corpus()
    for name, schema in schemas:
        body = chat_body(schema, name)
        ours = specs.json_schema(decode_chat(body).response_format.json_schema.schema)
        theirs = serialize(*key(params(json=json.loads(body)["response_format"]["json_schema"]["schema"])), compact,
                           False)
        assert ours == theirs, name
        assert json_schema_grammar(schema, "guidance", compact) == specs.json_schema(schema) == theirs, name
        assert json.loads(ours)["grammars"][0]["json_schema"]["x-guidance"] == {"whitespace_flexible": not compact}
    assert len(schemas) == 32
    theirs = serialize(*key(params(json_object=True)), compact, False)
    assert json_object_grammar("guidance", compact) == theirs == specs.json_object()


# -- (B) the tokenizer -------------------------------------------------------------------------------------------------


def test_lltokenizer_is_vllms(engine, hf, grammar_ns) -> None:
    from copy import copy as shallow

    from tokenizers import Tokenizer

    cfg, grammars = engine
    assert cfg.grammar_eos_token_ids == [EOS] and grammars.eos == {EOS}
    # the request's stop ids stay generation_config.json's (vLLM adds the tokenizer's EOS to its set; the engine
    # refuses to start without it: tests/structured/test_structured.py)
    assert cfg.eos_token_ids == [1, 50, 106]
    assert hf.eos_token == "<eos>" and hf.eos_token_id == EOS and len(hf) == VOCAB
    ours = grammars.backend.tokenizer
    theirs = vllm_ref.backend(grammar_ns, hf, cfg.vocab_size, True).ll_tokenizer
    for t in (ours, theirs):
        assert (t.vocab_size, t.eos_token, t.eos_tokens, t.is_canonical) == (VOCAB, EOS, [EOS], True)
    # what both are built from: the serialized tokenizer (32 MB), without padding and truncation
    hf_backend = shallow(hf.backend_tokenizer)
    hf_backend.no_padding()
    hf_backend.no_truncation()
    ours_json = Tokenizer.from_file(str(GEMMA / "tokenizer.json"))
    ours_json.no_padding()
    ours_json.no_truncation()
    same = hf_backend.to_str() == ours_json.to_str()
    assert same, "transformers serializes another tokenizer than tokenizer.json"
    ids = range(VOCAB)
    assert [i for i in ids if ours.decode_bytes([i]) != theirs.decode_bytes([i])] == []
    assert [i for i in ids if ours.is_special_token(i) != theirs.is_special_token(i)] == []
    assert {EOS, *SPECIAL} <= {i for i in ids if ours.is_special_token(i)}
    for text in ('{"n":1,"b":[{"i":0,"ge":["novel"],"la":null}]}', "Grüße, 東京 🙂 ::- -0.5e10"):
        assert ours.tokenize_str(text) == theirs.tokenize_str(text)


# -- (C) the masks -----------------------------------------------------------------------------------------------------


class Walk:
    """Random documents of a schema and the masks along them: ours (the engine's `Grammars`, the frontend's text) and
    vLLM's (its GuidanceBackend's grammar, its StructuredOutputManager's fill)."""

    def __init__(self, schema: dict, engine, grammar_ns: dict[str, Any], hf):
        cfg, self.grammars = engine
        self.compiled = self.grammars.compile(GrammarSpecs(backend="guidance", disable_any_whitespace=True)
                                              .json_schema(schema))
        self.key = grammar_ns["get_structured_output_key"](params(json=schema))
        self.vllm = vllm_ref.backend(grammar_ns, hf, cfg.vocab_size, True)
        self.manager = vllm_ref.manager(grammar_ns, self.vllm, 3)
        tok = self.grammars.backend.tokenizer
        # the walk's bias: tokens that close something (shorter documents)
        self.closing = np.array([any(s in tok.decode_bytes([i]) for s in (b"]", b"}", b",", b"null"))
                                 for i in range(VOCAB)])
        self.buf = np.zeros((4, words(VOCAB)), np.int32)
        self.positions = self.draft_masks = 0

    def check(self, ok: bool, what: str, detail: Any = "") -> None:
        if not ok:
            pytest.fail(f"{what}: {detail}", pytrace=False)

    def document(self, rng: np.random.Generator, limit: int = 2000) -> list[int]:
        """A random document the grammar accepts, then the EOS: each token one the mask allows, 3 in 4 times one
        that holds ], }, a comma or null when the mask has one."""
        m, toks = self.grammars.matcher(self.compiled), []
        while not m.is_stopped():
            self.grammars.fill([(m, 0)], [], self.buf, 1)
            allowed = unpack_mask(self.buf[0], VOCAB)
            closing = np.flatnonzero(allowed & self.closing)
            pick = closing if len(closing) and rng.random() < 0.75 else np.flatnonzero(allowed)
            t = int(rng.choice(pick))
            self.check(m.consume_tokens([t]) and len(toks) < limit, "a random document did not end", len(toks))
            toks.append(t)
        return [*toks, EOS]

    def replay(self, doc: list[int], rng: np.random.Generator) -> None:
        """Every position of `doc`: the single mask, the iff property, the first and last masks, and the 1 + 3 masks
        along the true next tokens and along a copy with one random draft, ours against vLLM's."""
        g, m = self.grammars, self.grammars.matcher(self.compiled)
        theirs = self.vllm.compile_grammar(*self.key)
        req = vllm_ref.request(theirs)
        for i, t in enumerate(doc):
            self.positions += 1
            g.fill([(m, 0)], [], self.buf, 1)
            mask = self.buf[0].copy()
            self.check(np.array_equal(mask, self.manager.grammar_bitmask({"r": req}, ["r"], {})[0]),
                       "a mask differs from vLLM's", i)
            allowed = set(np.flatnonzero(unpack_mask(mask, VOCAB)).tolist())
            ff, stopped = m.compute_ff_tokens(), m.is_stopped()
            self.check((len(allowed) == 1) == (bool(ff) or stopped), "popcount 1 iff forced or stopped", i)
            self.check(not ff or allowed == {ff[0]}, "a forced token's mask is that token", i)
            self.check(not stopped or (ff == [] and allowed == {EOS}), "a stopped grammar allows the EOS only", i)
            self.check(stopped == (t == EOS), "the EOS exactly after the closing brace", i)
            self.check(i >= len(FIRST) or allowed == FIRST[i], "the first masks", (i, sorted(allowed)))
            self.check(not allowed & set(SPECIAL), "a mask allows Gemma's other stop ids", i)
            true = doc[i:i + 3]
            wrong = list(true)
            wrong[int(rng.integers(len(wrong)))] = int(rng.integers(VOCAB))
            for drafts in (true, wrong):
                self.drafts(m, req, drafts, i)
            if t != EOS:
                self.check(m.consume_tokens([t]) and theirs.accept_tokens("r", [t]), "a token was rejected", i)

    def drafts(self, m: Any, req: Any, drafts: list[int], i: int) -> None:
        """Our 1 + k masks along `drafts` against vLLM's: its drafts cut by `validate_tokens`, then its fill (a mask,
        accept, ...; the bonus mask; rollback). Compared: the rows verification reads, up to the first draft the
        grammar rejects (that row decides the token there) and not past a drafted EOS (the request ends there)."""
        self.buf[:] = 0x5A5A5A5A  # any row compared must be written
        self.grammars.fill([], [(m, 0, list(drafts))], self.buf, 1 + len(drafts))
        valid = self.manager.validate_tokens(req, list(drafts))
        theirs = self.manager.grammar_bitmask({"r": req}, ["r"], {"r": valid} if valid else {}).copy()
        n = 1 + (valid.index(EOS) if EOS in valid else len(valid))
        k = self.manager.vllm_config.num_speculative_tokens  # vLLM fills a fixed block of k + 1 rows a request
        self.check(len(theirs) == 1 + k and np.array_equal(self.buf[:n], theirs[:n]),
                   "a draft mask differs from vLLM's", (i, drafts, valid))
        self.draft_masks += n


def test_masks_equal_vllms(engine, grammar_ns, hf) -> None:
    walk = Walk(json.loads(SYNTHETIC.read_text()), engine, grammar_ns, hf)
    for d in range(40):
        rng = np.random.default_rng(0x9A417 + d)
        walk.replay(walk.document(rng), rng)
    assert walk.positions > 1000 and walk.draft_masks > 3 * walk.positions, "too few masks compared"

