"""The tests' model fixtures: tokenizers, chat templates and configuration files from public Hugging Face
repositories (no weights), and the chat-template corpora rendered from them, in the git-ignored tests/.fixtures/:

    tests/.fixtures/models/<name>/      a repository's files, at a pinned commit
    tests/.fixtures/corpus/<name>.jsonl  500 seeded chat-template cases with transformers' rendering and token ids
                                         (tests/scripts/template_corpus.py), for tests/api/test_render.py and
                                         tests/api/test_output.py

    uv run tests/scripts/fetch_fixtures.py            # what is missing (about 120 MB downloaded)
    uv run tests/scripts/fetch_fixtures.py --check    # verify what is there; download nothing

Every file is checked against its sha256, the corpora too: a corpus that comes out different (another transformers)
fails here rather than in the tests. Downloads send no token. Tests that need a fixture skip without it and name this
script.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / ".fixtures"
MODELS = FIXTURES / "models"
CORPUS = FIXTURES / "corpus"

# name: (repository, commit, {file: sha256})
REPOS: dict[str, tuple[str, str, dict[str, str]]] = {
    "qwen3.8-27b": ("Qwen/Qwen3.8-27B-FP8", "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a", {
        "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        "config.json": "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc",
        "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    }),
    "qwen3-next-80b": ("Qwen/Qwen3-Next-80B-A3B-Instruct", "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7", {
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json": "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
    }),
    "gemma-4-26b-a4b": ("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52", {
        "chat_template.jinja": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        "tokenizer_config.json": "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633",
    }),
    "nemotron-3-nano-4b": ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f", {
        "chat_template.jinja": "ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862",
        "tokenizer.json": "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
        "tokenizer_config.json": "48de4056b0b17de26e03232fdc1f55b70595c9354ceb2ed061f724f45620aa41",
    }),
    "step-3.5-flash": ("stepfun-ai/Step-3.5-Flash", "ab446a3de5e171ea341227e24bb1f090e1b771f7", {
        "chat_template.jinja": "b18b99dff7261214c43e0402cba9f573e3240be6f9bb7ccaa54bb8df97e49dca",
        "tokenizer.json": "b564c620eb77fa11d0926011c2202347d6cfc358d79724ee04ae7007e13636f0",
        "tokenizer_config.json": "78202af487f4d4360e8d15cb0506d60718f8599770599c6f28f7f1fa045a591f",
    }),
    "gemma-4-31b-it": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", {
        "chat_template.jinja": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        "config.json": "e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1",
        "generation_config.json": "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
        "model.safetensors.index.json": "d4aff3b976d69c123a29d1c085d7ba4de1ac3f4ca1726a7f81e1b11462a64ea2",
        "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
        "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        "tokenizer_config.json": "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633",
    }),
}
# The corpora (500 cases, seed 20261002) as transformers 5.20.1 and tokenizers 0.23.3 render them (and 5.20.0, 0.23.2)
CORPORA: dict[str, str] = {
    "qwen3.8-27b": "0700396a8901489e282854d0ff13f731d1dce17c2d6307404d6b7b52d7dc6943",
    "qwen3-next-80b": "a774ff4836ca848eb42cd4d0d2efb887b35179a2e8fbd1ea575c24fbad67198f",
    "gemma-4-26b-a4b": "5aa67c0b67c7856f3d48a2cb33f3319e31fe2893ccba63a42908c3f22080ddbc",
    "nemotron-3-nano-4b": "f253a90b1cb6ff3805693fe02f368465fb9366b42b85a6182875eb464877e934",
    "step-3.5-flash": "d05a0ae8895b666c294ad010ef68d65f40208094901dcc2022dc22bfbf540907",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ok(path: Path, want: str) -> bool:
    return path.is_file() and sha256(path) == want


def download(repo: str, commit: str, name: str, dest: Path, want: str) -> None:
    """One file at a commit, without credentials, into `dest` once its hash checks."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{repo}/resolve/{commit}/{name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as r:
                while block := r.read(1 << 20):
                    tmp.write(block)
            tmp.close()
            if (got := sha256(Path(tmp.name))) != want:
                raise SystemExit(f"{repo}/{name} at {commit[:12]}: sha256 {got}, expected {want}")
            os.replace(tmp.name, dest)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def corpus(name: str, dest: Path, want: str) -> None:
    """A model's corpus, rendered by transformers (tests/scripts/template_corpus.py), offline, into `dest` once it
    checks."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env["HF_HUB_OFFLINE"] = "1"
    try:
        subprocess.run([sys.executable, str(ROOT / "tests" / "scripts" / "template_corpus.py"), "--model-dir",
                        str(MODELS / name), "--n", "500", "--out", str(tmp)], env=env, check=True)
        if (got := sha256(tmp)) != want:
            raise SystemExit(f"the {name} corpus: sha256 {got}, expected {want} (another transformers or tokenizers "
                             "renders it differently)")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify the fixtures; download nothing")
    args = ap.parse_args()
    missing = []
    for name, (repo, commit, files) in REPOS.items():
        for file, want in files.items():
            path = MODELS / name / file
            if ok(path, want):
                continue
            if args.check:
                missing.append(str(path.relative_to(ROOT)))
                continue
            print(f"{repo}/{file}", flush=True)
            download(repo, commit, file, path, want)
    for name, want in CORPORA.items():
        path = CORPUS / f"{name}.jsonl"
        if ok(path, want):
            continue
        if args.check:
            missing.append(str(path.relative_to(ROOT)))
            continue
        print(f"the {name} corpus", flush=True)
        corpus(name, path, want)
    if missing:
        raise SystemExit("missing or different: " + ", ".join(missing))
    print(f"the fixtures are in {FIXTURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
