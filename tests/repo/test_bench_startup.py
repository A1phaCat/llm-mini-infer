"""The start-up timer (bench/load/startup.py) against fake servers written into a temp folder: no GPU, no model."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ..support.paths import ROOT


def _load():
    name = "bench_load_startup"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench/load/startup.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load()

FAKE = textwrap.dedent(
    """
    import json, signal, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port, sleep_s, stubborn = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3] == "1"
    if stubborn:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(sleep_s)
    t_listen = time.monotonic()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200 if time.monotonic() - t_listen > 0.5 else 503, {})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._send(200, {"choices": [{"message": {"role": "assistant", "content": "hello"}}]})

    HTTPServer(("127.0.0.1", port), H).serve_forever()
    """
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake(tmp_path: Path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE)

    def make(sleep_s: float = 0.5, stubborn: bool = False) -> tuple[list[str], str]:
        port = free_port()
        return [sys.executable, str(script), str(port), str(sleep_s), "1" if stubborn else "0"], f"http://127.0.0.1:{port}"

    return make


def test_ready_and_first_answer_follow_the_start_delay(fake, tmp_path) -> None:
    argv, base = fake(sleep_s=1.0)
    h = st.launch(argv, env=None, cwd="/", log_path=tmp_path / "s.log", base_url=base)
    try:
        ready = st.wait_ready(h, "/health", timeout=30)
        answer = st.first_answer(h, "m", timeout=30)
    finally:
        stopped = st.stop(h)
    assert ready >= 1.5
    assert answer >= ready
    assert stopped["port_free"] is True
    assert stopped["signal"] == "SIGINT"


def test_the_cli_writes_the_report_for_two_starts(fake, tmp_path) -> None:
    argv, base = fake(sleep_s=0.2)
    out = tmp_path / "r.json"
    cmd = [sys.executable, str(ROOT / "bench/load/startup.py"), "--out", str(out), "--base-url", base, "--model", "m",
           "--starts", "2", "--timeout", "30", "--log", str(tmp_path / "l.log"), "--", *argv]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=55)
    assert res.returncode == 0, res.stderr
    rep = json.loads(out.read_text())
    assert {"argv", "cwd", "base_url", "ready_path", "model", "env_names", "wiped", "evicted", "starts"} <= set(rep)
    assert rep["argv"][0] == Path(sys.executable).name
    assert rep["env_names"] == sorted(rep["env_names"])
    assert len(rep["starts"]) == 2
    for s in rep["starts"]:
        assert s["error"] is None
        assert s["first_answer_s"] >= s["ready_s"] >= 0.7
        assert s["stopped"]["port_free"] is True


def test_a_child_that_exits_early_is_a_start_error_with_its_log(tmp_path) -> None:
    h = st.launch([sys.executable, "-c", "import sys; print('boom line'); sys.exit(3)"], env=None, cwd="/",
                  log_path=tmp_path / "e.log", base_url=f"http://127.0.0.1:{free_port()}")
    try:
        with pytest.raises(st.StartError) as ei:
            st.wait_ready(h, timeout=30)
    finally:
        st.stop(h)
    assert ei.value.exit_code == 3
    assert "boom line" in ei.value.log_tail


def test_a_child_that_ignores_signals_is_killed_and_the_port_comes_free(fake, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(st, "STOP_WAITS", (0.5, 0.5, 5.0))
    argv, base = fake(sleep_s=0.1, stubborn=True)
    h = st.launch(argv, env=None, cwd="/", log_path=tmp_path / "k.log", base_url=base)
    try:
        st.wait_ready(h, timeout=30)
    finally:
        stopped = st.stop(h)
    assert stopped["signal"] == "SIGKILL"
    assert stopped["port_free"] is True
    assert h.proc.poll() is not None


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False  # macOS answers EPERM for a group that holds only zombies awaiting reaping
    return True


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="no process groups on this platform")
def test_stop_kills_group_members_left_behind_by_an_exited_leader(tmp_path) -> None:
    code = ("import subprocess, sys; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            "print(p.pid, flush=True)")
    h = st.launch([sys.executable, "-c", code], env=None, cwd="/", log_path=tmp_path / "g.log",
                  base_url=f"http://127.0.0.1:{free_port()}")
    pgid = h.proc.pid
    try:
        h.proc.wait(timeout=30)  # the leader exits at once; its child keeps the group alive
        os.killpg(pgid, 0)
        st.stop(h)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _group_alive(pgid):
            time.sleep(0.05)
        assert not _group_alive(pgid)
    finally:
        with contextlib.suppress(OSError):
            os.killpg(pgid, signal.SIGKILL)


def test_a_client_error_on_the_first_request_names_the_status_and_type(tmp_path) -> None:
    script = tmp_path / "bad.py"
    script.write_text(textwrap.dedent(
        """
        import json, sys
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                data = json.dumps({"error": {"type": "not_found", "message": "secret detail"}}).encode()
                self.send_response(404); self.send_header("Content-Length", str(len(data))); self.end_headers()
                self.wfile.write(data)

        HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
        """
    ))
    port = free_port()
    h = st.launch([sys.executable, str(script), str(port)], env=None, cwd="/", log_path=tmp_path / "b.log",
                  base_url=f"http://127.0.0.1:{port}")
    try:
        st.wait_ready(h, timeout=30)
        with pytest.raises(st.StartError) as ei:
            st.first_answer(h, "m", timeout=30)
    finally:
        st.stop(h)
    assert "404" in str(ei.value) and "not_found" in str(ei.value)
    assert "secret detail" not in str(ei.value)


def test_wipe_refuses_weights_and_empties_an_ordinary_folder(tmp_path) -> None:
    ok = tmp_path / "cache"
    (ok / "sub").mkdir(parents=True)
    (ok / "sub" / "a.txt").write_text("x")
    (ok / "b.txt").write_text("y")
    with_weights = tmp_path / "w"
    (with_weights / "deep").mkdir(parents=True)
    (with_weights / "deep" / "m.safetensors").write_text("z")
    hf = tmp_path / "hf"
    (hf / "models--x").mkdir(parents=True)
    with pytest.raises(ValueError):
        st.wipe([str(ok), str(with_weights)])
    assert (ok / "b.txt").exists()
    with pytest.raises(ValueError):
        st.wipe([str(hf)])
    assert (hf / "models--x").exists()
    res = st.wipe([str(ok), str(tmp_path / "new")])
    assert list(ok.iterdir()) == []
    assert (tmp_path / "new").is_dir()
    assert res[0]["removed"] == 2


def test_wipe_refuses_the_root_the_home_and_a_parent_of_the_cwd(tmp_path) -> None:
    for d in ("/", str(Path.home()), str(Path.cwd().parent)):
        with pytest.raises(ValueError):
            st.wipe([d])


def test_the_cli_exits_2_on_a_refused_wipe(tmp_path) -> None:
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "x.gguf").write_text("z")
    out = tmp_path / "r.json"
    res = subprocess.run(
        [sys.executable, str(ROOT / "bench/load/startup.py"), "--out", str(out), "--model", "m", "--wipe",
         str(tmp_path / "w"), "--", sys.executable, "-c", "pass"], capture_output=True, text=True, timeout=30)
    assert res.returncode == 2
    assert (tmp_path / "w" / "x.gguf").exists()
    assert not out.exists()


@pytest.mark.skipif(not hasattr(__import__("os"), "posix_fadvise"), reason="no posix_fadvise on this platform")
def test_evict_records_files_and_bytes(tmp_path) -> None:
    (tmp_path / "f.bin").write_bytes(b"x" * 1000)
    (tmp_path / "link.bin").symlink_to(tmp_path / "f.bin")
    rec = st.evict([str(tmp_path)])[0]
    assert rec["files"] == 1
    assert rec["bytes"] == 1000
    assert rec["errors"] == 0
    assert {"resident_before", "resident_after"} <= set(rec)


def test_redact_hides_secrets_and_cuts_the_executable() -> None:
    argv = ["/opt/venv/bin/vllm", "serve", "m", "--api-key", "sekret", "--hf-token=tok", "--token", "t2", "--port", "1"]
    assert st.redact(argv) == ["vllm", "serve", "m", "--api-key", "<redacted>", "--hf-token=<redacted>", "--token",
                               "<redacted>", "--port", "1"]
    assert argv[3 + 1] == "sekret"
