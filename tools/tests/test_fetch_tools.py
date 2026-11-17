"""The pinned command-line tools (tools/fetch_tools.py): every download is checked against its pinned sha256 before
anything lands in tools/bin/, an archive's binary (.tar.gz, .zip) is unpacked from its pinned member, and a copy that
is another version, or was changed after the fetch, is not used. Offline: the downloads are served from memory."""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import fetch_tools  # noqa: E402

BINARY = b"#!/bin/sh\necho fake tool\n"


def tar_with(member: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(member)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def zip_with(member: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(member, data)
    return buf.getvalue()


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """tools/bin/ in a temporary folder, three pinned tools (a bare binary, a .tar.gz, a .zip), downloads served from a
    dict."""
    served: dict[str, bytes] = {"https://x/raw": BINARY, "https://x/tool.tar.gz": tar_with("dir/tool", BINARY),
                                "https://x/zipped.zip": zip_with("dir/bin/zipped", BINARY)}
    sha = {url: hashlib.sha256(data).hexdigest() for url, data in served.items()}
    pins = {"raw": {"version": "v1", "darwin-arm64": ("https://x/raw", sha["https://x/raw"], None)},
            "tool": {"version": "v2",
                     "darwin-arm64": ("https://x/tool.tar.gz", sha["https://x/tool.tar.gz"], "dir/tool")},
            "zipped": {"version": "v3",
                       "darwin-arm64": ("https://x/zipped.zip", sha["https://x/zipped.zip"], "dir/bin/zipped")}}
    monkeypatch.setattr(fetch_tools, "BIN", tmp_path / "bin")
    monkeypatch.setattr(fetch_tools, "PINS", pins)
    monkeypatch.setattr(fetch_tools, "this_platform", lambda: "darwin-arm64")
    monkeypatch.setattr(fetch_tools.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(served[req.full_url]))
    return served


def test_a_download_is_checked_then_unpacked(fake) -> None:
    for name in ("raw", "tool", "zipped"):
        assert fetch_tools.status(name) == "absent" and fetch_tools.path(name) is None
        exe = fetch_tools.fetch(name, log=lambda m: None)
        assert exe.read_bytes() == BINARY and exe.stat().st_mode & 0o111
        assert fetch_tools.status(name) == "ok" and fetch_tools.path(name) == exe
        assert exe.parent == fetch_tools.BIN / "darwin-arm64"


def test_a_download_that_is_not_the_pinned_one_installs_nothing(fake) -> None:
    fake["https://x/raw"] = BINARY + b"tampered"
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        fetch_tools.fetch("raw", log=lambda m: None)
    assert fetch_tools.status("raw") == "absent"
    assert not [p for p in (fetch_tools.BIN / "darwin-arm64").iterdir()]  # no binary, no stamp, no leftovers


def test_another_version_or_a_changed_binary_is_not_used(fake, monkeypatch) -> None:
    exe = fetch_tools.fetch("raw", log=lambda m: None)
    exe.write_bytes(BINARY + b"changed")
    assert fetch_tools.status("raw") == "modified" and fetch_tools.path("raw") is None
    fetch_tools.fetch("raw", log=lambda m: None)
    assert fetch_tools.status("raw") == "ok"
    monkeypatch.setitem(fetch_tools.PINS, "raw", {**fetch_tools.PINS["raw"], "version": "v9"})
    assert fetch_tools.status("raw") == "stale" and fetch_tools.path("raw") is None


def test_the_pins_cover_both_platforms_with_sha256s() -> None:
    for name, pin in fetch_tools.PINS.items():
        assert pin["version"].startswith("v"), name
        for plat in fetch_tools.PLATFORMS:
            url, sha, _member = pin[plat]
            assert url.startswith("https://") and pin["version"] in url, (name, plat)
            assert len(sha) == 64 and int(sha, 16) >= 0, (name, plat)


def test_a_platform_without_pins_is_stale_not_an_error(fake) -> None:
    folder = fetch_tools.BIN / "linux-riscv"
    folder.mkdir(parents=True)
    (folder / "raw").write_bytes(BINARY)
    fetch_tools._stamp("raw", "linux-riscv").write_text('{"version": "v1"}')
    assert fetch_tools.status("raw", "linux-riscv") == "stale"
