"""The reports' check (bench/reports/check.py): each rule flags a hand-made file, a clean report passes, and the
published reports in the repository have no findings."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ..support.paths import ROOT


def _load():
    name = "bench_reports_check"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench/reports/check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


check = _load()

REPORTS = ROOT / "bench" / "reports"


def _rules(path: Path) -> set[str]:
    return {f.rule for f in check.check_file(path)}


def _text(tmp_path: Path, content: str, name: str = "notes.txt") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_published_reports_have_no_findings():
    dirs = [d for d in sorted(REPORTS.iterdir()) if d.is_dir() and d.name != "__pycache__"]
    for d in dirs:
        found = check.check_tree(d)
        assert not found, [f.format(ROOT) for f in found]


@pytest.mark.parametrize(
    ("rule", "line"),
    [
        ("path", "log at /" + "Users/someone/work/run.log"),
        ("path", "cache in /" + "tmp/x"),
        ("path", "dir C:" + "\\Users\\x"),
        ("ip", "server 10.0.12.7 answered"),
        ("ip", "peer 2001:db8::7 up"),
        ("ip", "peer " + ":".join(["fe80", "1", "2", "3", "4", "5", "6", "7"])),
        ("host", "pod host abc-8000.proxy.runpod" + ".net"),
        ("host", "node.cluster.local"),
        ("host", "reach api.example-lab.com now"),
        ("pod", "pod " + "abc123" + "def456gh" + " is slow"),
        ("token", "key " + "hf_" + "A" * 30),
        ("token", "ghp_" + "b" * 25),
        ("token", "Authorization: Bearer " + "x" * 24),
        ("token", "api_key = " + "z" * 16),
        ("short-id", "commit " + "1a2b" + "3c4"),
        ("date", "run on " + "2026" + "-11-09"),
        ("date", "stamp " + "2026" + "-11-09T10:11:12Z"),
    ],
)
def test_text_rule_flags(tmp_path, rule, line):
    found = check.check_file(_text(tmp_path, "first line\n" + line + "\n"))
    assert {f.rule for f in found} == {rule}, [(f.rule, f.where) for f in found]
    assert found[0].where == "line 2"


def test_token_finding_never_prints_the_text(tmp_path):
    secret = "hf_" + "Q" * 30
    p = _text(tmp_path, f"x = {secret}\n")
    out = [f.format() for f in check.check_file(p)]
    assert out and all(secret not in line and "Q" * 10 not in line for line in out)
    j = tmp_path / "a.json"
    j.write_text(json.dumps({"k": secret}))
    assert all(secret not in f.format() for f in check.check_file(j))


def test_json_walks_keys_and_values_with_paths(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"a": {"b": ["fine", "see /" + "home/x/y"]}, "10.1.2.3": 1}))
    found = {(f.rule, f.where) for f in check.check_file(p)}
    assert ("path", "$.a.b[1]") in found
    assert ("ip", "$.10.1.2.3 (key)") in found


def test_json_key_value_secret(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"password": "p" * 16}))
    assert _rules(p) == {"token"}


def test_when_must_be_year_month(tmp_path):
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"when": "2026-11"}))
    assert not check.check_file(ok)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"when": "November"}))
    assert [(f.rule, f.where) for f in check.check_file(bad)] == [("date", "$.when")]


def test_short_id_allowed_under_answers_only(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"answers": {"q1": ["1a2b3" + "c4d5e", "9f8e" + "7d6"]}, "other": "9f8e" + "7d6"}))
    found = check.check_file(p)
    assert [(f.rule, f.where) for f in found] == [("short-id", "$.other")]


def test_clean_report_passes(tmp_path):
    report = {
        "schema": 3,
        "when": "2026-11",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "image_digest": "sha256:" + "ab12" * 16,
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "bind": "0.0.0.0",
        "loopback": "::1",
        "model": "huggingface.co/RedHatAI/some-model-FP8",
        "docs": "https://docs.vllm.ai/en/latest/ and https://objects.githubusercontent.com/x",
        "script": "bench/load/run.py",
        "version": "2.14.1+cu130",
        "module": "llm_mini_infer.hostcheck",
        "clock": "took 12:34:56 in total",
        "loads": [{"rate": 4.0, "answers": {"a": "1a2b3c4d5e6f7a8b"}}],
    }
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    assert check.check_file(p) == []
    md = _text(tmp_path, "Run `python3 bench/load/run.py --url http://localhost:8000` on 2026-11 data.\n", "README.md")
    assert check.check_file(md) == []


def _manifest(tmp_path: Path, data: object) -> Path:
    d = tmp_path / "manifests" / "model" / "family"
    d.mkdir(parents=True)
    p = d / "arm.json"
    p.write_text(json.dumps(data))
    return p


def test_manifest_known_keys_pass(tmp_path):
    p = _manifest(tmp_path, {"schema": 1, "arm": "a", "resolved": {"kv_tokens": 5}, "client": {"seed": 1}})
    assert check.check_file(p, tmp_path) == []


def test_manifest_unknown_top_level_key(tmp_path):
    p = _manifest(tmp_path, {"schema": 1, "hostname": "x"})
    assert [(f.rule, f.where) for f in check.check_file(p, tmp_path)] == [("manifest-key", "$.hostname")]


def test_manifest_unknown_nested_key(tmp_path):
    p = _manifest(tmp_path, {"schema": 1, "resolved": {"kv_tokens": 5, "secret_sauce": 1}})
    assert [(f.rule, f.where) for f in check.check_file(p, tmp_path)] == [("manifest-key", "$.resolved.secret_sauce")]


def test_json_outside_manifests_has_no_key_rule(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"anything": 1}))
    assert check.check_file(p, tmp_path) == []


def test_unreadable_files_are_findings(tmp_path):
    b = tmp_path / "blob.bin"
    b.write_bytes(b"\xff\xfe\x00\x80")
    assert _rules(b) == {"encoding"}
    j = tmp_path / "broken.json"
    j.write_text("{not json")
    assert _rules(j) == {"json"}


def test_main_reports_and_exit_codes(tmp_path, capsys):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "a.txt").write_text("nothing here\n")
    assert check.main([str(clean)]) == 0
    assert capsys.readouterr().out.strip() == "1 files checked, no findings"
    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "a.txt").write_text("at /" + "root/x\n")
    assert check.main([str(dirty)]) == 1
    out = capsys.readouterr().out
    assert out.strip() == "dirty/a.txt: path: line 1"
