"""A fake Hugging Face Hub for the download tests: the real huggingface_hub library talks to it through `HF_ENDPOINT`,
so the tests exercise its cache, offline mode, revisions and token header without the network.

It serves one model repository per name: the revision lookup (`/api/models/<repo>/revision/<rev>`), the file tree
(`/api/models/<repo>/tree/<commit>`) and the files (`/<repo>/resolve/<commit>/<file>`, HEAD and GET), and records
every request (method, path, and a hash of its Authorization header: never a token itself, a real one included: the
library sends a token it finds in the environment) so a test can tell what was downloaded and whether a token was sent.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse


def auth_digest(header: str) -> str:
    """What the hub records of an Authorization header."""
    return hashlib.sha256(header.encode()).hexdigest()[:16]


class Repo:
    def __init__(self, files: dict[str, bytes], commit: str | None = None, token: str | None = None):
        self.files = files
        self.commit = commit or hashlib.sha1(json.dumps(sorted(files)).encode()).hexdigest()
        self.token = token  # when set, every request needs `Authorization: Bearer <token>` (a gated repo)


class FakeHub:
    def __init__(self, repos: dict[str, Repo]):
        self.repos = repos
        self.requests: list[tuple[str, str, str | None]] = []  # (method, path, auth_digest(Authorization) or None)
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def do_HEAD(self) -> None:
                hub.handle(self, head=True)

            def do_GET(self) -> None:
                hub.handle(self, head=False)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> FakeHub:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()

    def downloads(self) -> list[str]:
        """The files whose contents were fetched (GETs of resolve URLs)."""
        return [p.split("/resolve/", 1)[1].split("/", 1)[1] for m, p, _ in self.requests
                if m == "GET" and "/resolve/" in p]

    # -- the handler -------------------------------------------------------------------------------------------------

    def handle(self, h: BaseHTTPRequestHandler, head: bool) -> None:
        url = urlparse(h.path)
        path = unquote(url.path)
        auth = h.headers.get("authorization")
        self.requests.append((h.command, path, auth_digest(auth) if auth else None))
        try:
            status, headers, body = self.route(path, auth)
        except (KeyError, ValueError):
            status, headers, body = 404, {"x-error-code": "RepoNotFound"}, b'{"error": "Repository not found"}'
        h.send_response(status)
        for k, v in headers.items():
            h.send_header(k, v)
        h.send_header("content-length", str(len(body)))
        h.end_headers()
        if not head:
            h.wfile.write(body)

    def route(self, path: str, auth: str | None) -> tuple[int, dict[str, str], bytes]:
        js = {"content-type": "application/json"}
        if path.startswith("/api/models/"):
            rest = path[len("/api/models/"):]
            org, name, kind, rev = rest.split("/", 3)
            repo = self.repos[f"{org}/{name}"]
            if repo.token and auth != f"Bearer {repo.token}":
                return 401, {**js, "x-error-code": "GatedRepo"}, b'{"error": "Access to this repo is restricted"}'
            if kind == "revision":
                if rev not in ("main", repo.commit):
                    return 404, {**js, "x-error-code": "RevisionNotFound"}, b'{"error": "Invalid rev id"}'
                return 200, js, json.dumps({"id": f"{org}/{name}", "modelId": f"{org}/{name}", "sha": repo.commit,
                                            "private": False, "siblings": [{"rfilename": f} for f in repo.files]
                                            }).encode()
            if kind == "tree":
                tree = [{"type": "file", "path": f, "size": len(data), "oid": hashlib.sha1(data).hexdigest()}
                        for f, data in sorted(repo.files.items())]
                return 200, js, json.dumps(tree).encode()
            raise KeyError(path)
        org, name, kind, rev, file = path.lstrip("/").split("/", 4)
        repo = self.repos[f"{org}/{name}"]
        if kind != "resolve" or rev not in ("main", repo.commit):
            raise KeyError(path)
        if repo.token and auth != f"Bearer {repo.token}":
            return 401, {"x-error-code": "GatedRepo"}, b"Access to this repo is restricted"
        data = repo.files[file]
        return 200, {"x-repo-commit": repo.commit, "etag": f'"{hashlib.sha1(data).hexdigest()}"',
                     "content-type": "application/octet-stream"}, data


def main() -> None:
    """`python3 fake_hub.py <repo id> <directory> [port]`: serve a directory's files as one repository, so that a
    server elsewhere (a container, a kind cluster) downloads its model by repo id from it, as from Hugging Face."""
    import sys
    from pathlib import Path

    repo, folder = sys.argv[1], Path(sys.argv[2])
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    hub = FakeHub({repo: Repo({p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()})})
    hub.server.server_close()
    hub.server = ThreadingHTTPServer(("0.0.0.0", port), hub.server.RequestHandlerClass)
    print(f"fake hub: {repo} ({len(hub.repos[repo].files)} files, commit {hub.repos[repo].commit}) on port {port}",
          flush=True)
    hub.server.serve_forever()


if __name__ == "__main__":
    main()
