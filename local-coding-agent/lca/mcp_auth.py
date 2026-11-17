"""OAuth sign-in for HTTP MCP servers (MCP authorization spec: OAuth 2.1 with PKCE, discovery, dynamic registration).

    lca mcp --login <server>       (or "Sign in" under /mcp in VS Code)

1. The server answers 401 with `WWW-Authenticate: Bearer resource_metadata="..."`.
2. Its protected-resource metadata names the authorization server, whose metadata gives the endpoints.
3. If the server allows dynamic client registration, the agent registers itself; otherwise set
   `oauth = { client_id = "...", scopes = ["..."] }` on the server in the config.
4. Your browser opens the sign-in page; the result comes back to a one-time listener on 127.0.0.1.
5. Tokens are kept in ~/.lca/mcp-auth/ (readable only by you) and refreshed automatically when they expire.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from lca import __version__
from lca.config import lca_home

LOGIN_TIMEOUT_S = 300


class AuthError(Exception):
    pass


@dataclass
class Tokens:
    access_token: str
    token_endpoint: str
    client_id: str
    refresh_token: str = ""
    expires_at: float = 0.0
    client_secret: str = ""
    resource: str = ""

    def expired(self) -> bool:
        return bool(self.expires_at) and time.time() > self.expires_at - 60


def _token_file(server_url: str) -> Path:
    return lca_home() / "mcp-auth" / f"{hashlib.sha256(server_url.encode()).hexdigest()[:16]}.json"


def load_tokens(server_url: str) -> Tokens | None:
    try:
        data = json.loads(_token_file(server_url).read_text(encoding="utf-8"))
        return Tokens(**{k: v for k, v in data.items() if k in Tokens.__dataclass_fields__})
    except (OSError, ValueError, TypeError):
        return None


def save_tokens(server_url: str, tokens: Tokens) -> None:
    path = _token_file(server_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"server": server_url, **asdict(tokens)}, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def forget_tokens(server_url: str) -> None:
    _token_file(server_url).unlink(missing_ok=True)


def _post_token(client: httpx.Client, tokens_endpoint: str, data: dict[str, str], client_secret: str = "") -> dict[str, Any]:
    if client_secret:
        data = {**data, "client_secret": client_secret}
    r = client.post(tokens_endpoint, data=data, headers={"Accept": "application/json"})
    if r.status_code >= 400:
        raise AuthError(f"the token endpoint answered HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    if "access_token" not in body:
        raise AuthError(f"no access_token in the token response: {str(body)[:300]}")
    return body


def _tokens_from(body: dict[str, Any], old: Tokens | None, endpoint: str, client_id: str, client_secret: str, resource: str) -> Tokens:
    expires_in = body.get("expires_in")
    return Tokens(
        access_token=body["access_token"], token_endpoint=endpoint, client_id=client_id, client_secret=client_secret,
        refresh_token=body.get("refresh_token") or (old.refresh_token if old else ""),
        expires_at=time.time() + float(expires_in) if expires_in else 0.0, resource=resource,
    )


def refresh_tokens(server_url: str, tokens: Tokens) -> Tokens | None:
    """A new access token from the refresh token, or None when that is not possible (sign in again)."""
    if not tokens.refresh_token:
        return None
    try:
        with httpx.Client(timeout=30) as client:
            data = {"grant_type": "refresh_token", "refresh_token": tokens.refresh_token, "client_id": tokens.client_id}
            if tokens.resource:
                data["resource"] = tokens.resource
            body = _post_token(client, tokens.token_endpoint, data, tokens.client_secret)
    except (httpx.HTTPError, AuthError, ValueError):
        return None
    new = _tokens_from(body, tokens, tokens.token_endpoint, tokens.client_id, tokens.client_secret, tokens.resource)
    save_tokens(server_url, new)
    return new


def access_token(server_url: str, force_refresh: bool = False) -> str | None:
    """The stored access token for a server, refreshed when it expired (or when forced after a 401)."""
    tokens = load_tokens(server_url)
    if tokens is None:
        return None
    if force_refresh or tokens.expired():
        tokens = refresh_tokens(server_url, tokens)
    return tokens.access_token if tokens else None


# ---------------------------------------------------------------- discovery


def _well_known(base: str, name: str) -> list[str]:
    """RFC 8414 / 9728 locations: the well-known path inserted before the URL's path, then at the root."""
    u = urlparse(base)
    origin = f"{u.scheme}://{u.netloc}"
    path = u.path.rstrip("/")
    urls = [f"{origin}/.well-known/{name}{path}"] if path else []
    return urls + [f"{origin}/.well-known/{name}"]


def _get_json(client: httpx.Client, urls: list[str]) -> dict[str, Any] | None:
    for url in urls:
        try:
            r = client.get(url, headers={"Accept": "application/json", "MCP-Protocol-Version": "2025-06-18"})
        except httpx.HTTPError:
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                continue
    return None


def discover(client: httpx.Client, server_url: str) -> dict[str, Any]:
    probe = client.post(server_url, json={"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "local-coding-agent", "version": __version__}}},
        headers={"Accept": "application/json, text/event-stream"})
    if probe.status_code != 401:
        raise AuthError(f"the server did not ask for sign-in (HTTP {probe.status_code}); no login is needed.")
    challenge = probe.headers.get("www-authenticate", "")
    m = re.search(r'resource_metadata="([^"]+)"', challenge)
    prm = _get_json(client, [m.group(1)] if m else _well_known(server_url, "oauth-protected-resource")) or {}
    issuer = (prm.get("authorization_servers") or [f"{urlparse(server_url).scheme}://{urlparse(server_url).netloc}"])[0]
    meta = _get_json(client, _well_known(issuer, "oauth-authorization-server") + _well_known(issuer, "openid-configuration"))
    if not meta or not meta.get("authorization_endpoint") or not meta.get("token_endpoint"):
        raise AuthError(f"could not find the authorization server metadata for {issuer}.")
    scope = re.search(r'scope="([^"]+)"', challenge)
    return {
        **meta, "resource": prm.get("resource") or server_url,
        "challenge_scopes": scope.group(1).split() if scope else prm.get("scopes_supported") or [],
    }


# ---------------------------------------------------------------- the browser round trip


class _Callback(BaseHTTPRequestHandler):
    result: dict[str, str] = {}
    done: threading.Event

    def log_message(self, *a: Any) -> None:
        pass

    def do_GET(self) -> None:
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        if "code" not in q and "error" not in q:
            self.send_response(404)
            self.end_headers()
            return
        type(self).result.update(q)
        ok = "code" in q
        body = (f"<html><body style='font-family:sans-serif'><h3>{'Signed in' if ok else 'Sign-in failed'}</h3>"
                "<p>You can close this window and return to the agent.</p></body></html>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        type(self).done.set()


def login(
    name: str, server_url: str, open_url: Callable[[str], Any], oauth: dict[str, Any] | None = None,
    timeout_s: float = LOGIN_TIMEOUT_S, cancelled: Callable[[], bool] = lambda: False,
) -> Tokens:
    """Sign in to an MCP server in the browser and store the tokens."""
    oauth = oauth or {}
    handler = type("Callback", (_Callback,), {"result": {}, "done": threading.Event()})
    listener = HTTPServer(("127.0.0.1", 0), handler)
    redirect_uri = f"http://127.0.0.1:{listener.server_port}/callback"
    threading.Thread(target=listener.serve_forever, daemon=True).start()
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            meta = discover(client, server_url)
            client_id, client_secret = str(oauth.get("client_id", "")), str(oauth.get("client_secret", ""))
            if not client_id:
                if not meta.get("registration_endpoint"):
                    raise AuthError(f"{name} does not support automatic client registration. Register an app with the "
                                    f"provider and set oauth = {{ client_id = \"...\" }} on [mcp.servers.{name}].")
                r = client.post(meta["registration_endpoint"], json={
                    "client_name": "local-coding-agent", "redirect_uris": [redirect_uri], "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
                })
                if r.status_code >= 400:
                    raise AuthError(f"client registration failed (HTTP {r.status_code}): {r.text[:300]}")
                reg = r.json()
                client_id, client_secret = reg["client_id"], reg.get("client_secret", "")
            verifier = secrets.token_urlsafe(48)
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            state = secrets.token_urlsafe(24)
            scopes = oauth.get("scopes") or meta.get("challenge_scopes") or []
            params = {
                "response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri, "state": state,
                "code_challenge": challenge, "code_challenge_method": "S256", "resource": meta["resource"],
                **({"scope": " ".join(scopes)} if scopes else {}),
            }
            sep = "&" if "?" in meta["authorization_endpoint"] else "?"
            open_url(meta["authorization_endpoint"] + sep + urlencode(params))
            deadline = time.monotonic() + timeout_s
            while not handler.done.wait(0.25):
                if cancelled() or time.monotonic() > deadline:
                    raise AuthError("sign-in was not completed in time." if not cancelled() else "sign-in was cancelled.")
            result = handler.result
            if "error" in result:
                raise AuthError(f"the sign-in was refused: {result.get('error_description') or result['error']}")
            if result.get("state") != state:
                raise AuthError("the sign-in response did not match this request (state mismatch); try again.")
            body = _post_token(client, meta["token_endpoint"], {
                "grant_type": "authorization_code", "code": result["code"], "redirect_uri": redirect_uri,
                "client_id": client_id, "code_verifier": verifier, "resource": meta["resource"],
            }, client_secret)
    except httpx.HTTPError as e:
        raise AuthError(f"sign-in failed: {e}") from e
    finally:
        listener.shutdown()
    tokens = _tokens_from(body, None, meta["token_endpoint"], client_id, client_secret, meta["resource"])
    save_tokens(server_url, tokens)
    return tokens
