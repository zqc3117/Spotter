"""HTTP-JSON RPC transport for the recovery_explore env service.

COPY-AND-ADAPT of the vendored RPent RPC layer (Apache-2.0, see
``vendor/rpent/LICENSE``; upstream commit in ``vendor/rpent/UPSTREAM_COMMIT.txt``).
Sources, in the order they appear below:

* ``vendor/rpent/utils/rpc/rpc_client.py``      -> ``RpcError`` / ``check_response`` / ``RpcClient``
* ``vendor/rpent/utils/rpc/http_rpc.py``        -> ``_from_json`` / ``_NumpyEncoder`` /
                                                   ``HttpRpcClient`` / ``HttpRpcServer``
* ``vendor/rpent/utils/rpc/rpc_facade.py``      -> ``RpcFacade``
* ``vendor/rpent/utils/rpc/main_thread_serve.py`` -> ``MainThreadServeMixin``
* ``vendor/rpent/utils/rpc/client_utils.py``    -> ``parse_endpoint`` / ``wait_for_ready``

WHY a copy instead of an import: the vendored files import ``rpent.utils.logging``,
``rpent.utils.rwlock``, ``rpent.utils.daemon`` and ``rpent.utils.egl``, none of which
were vendored (only ``utils/rpc/*`` was), and ``rpent`` is deliberately NOT installed
into the Cosmos env (its pip extra wants a robosuite fork + robocasa365, which would
break the official env). So the four
missing helpers are reimplemented here, minimally, and everything else is kept
structurally identical to upstream so a future re-vendor is a diff, not a rewrite.

WHAT WAS DROPPED relative to upstream, and why:

* sessions (``enable_sessions`` / ``session.register`` / idle sweep): this service is
  ONE process holding ONE env with a single driver; there is no per-client policy
  state to isolate.
* the ``RWLock`` read/write dispatch split: ``MainThreadServeMixin`` already funnels
  every call through one thread (see below), so the lock could never be contended.

WHAT WAS KEPT VERBATIM, and why it matters:

* ``MainThreadServeMixin``: MuJoCo's EGL context is THREAD-BOUND. The HTTP server is
  a ThreadingHTTPServer, so without this mixin a render served from a worker thread
  hits a context that belongs to another thread and the process dies. Every dispatch
  is therefore executed on the thread that called ``serve`` (the process main thread).
* the ndarray-as-base64 JSON tagging: ``tolist()`` stringifies every element, which is
  unusable for 512x512x3 frames and HxW float depth maps.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Literal

import numpy as np

DEFAULT_TIMEOUT_S = 30.0
_DIRECT_HOSTS = frozenset({"127.0.0.1", "localhost"})

logger = logging.getLogger("recovery_explore.rpc")


# ---------------------------------------------------------------------------
#  Local reimplementations of the rpent.utils helpers that were not vendored
# ---------------------------------------------------------------------------


def watch_parent_death(on_death: Callable[[], None]) -> None:
    """Fire ``on_death`` when stdin closes, i.e. when the launching parent dies.

    Reimplementation of ``rpent.utils.daemon.watch_parent_death`` (not vendored).
    The parent keeps a pipe open on our stdin; EOF means it went away, and an
    orphaned env process is a leaked GPU context, so we shut down.
    """

    def _wait() -> None:
        try:
            while sys.stdin.readline():
                pass
        except Exception:  # pragma: no cover - parent died mid-read
            pass
        on_death()

    threading.Thread(target=_wait, daemon=True, name="parent-watch").start()


def configure_egl_device(device_id: int) -> None:
    """Pin MuJoCo's offscreen renderer to one physical GPU.

    Reimplementation of ``rpent.utils.egl.configure_egl_device`` (not vendored).
    Must run BEFORE mujoco/robosuite is imported -- the EGL device is read at
    context creation. See ``serve_main``'s comment for why we deliberately do
    NOT set ``CUDA_VISIBLE_DEVICES`` here.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(device_id))


# ---------------------------------------------------------------------------
#  Client protocol (vendor/rpent/utils/rpc/rpc_client.py)
# ---------------------------------------------------------------------------


class RpcError(RuntimeError):
    """Raised when a remote method call returns an error."""

    def __init__(self, method: str, message: str, *, traceback: str | None = None):
        super().__init__(f"{method}: {message}")
        self.method = method
        self.server_traceback = traceback


def check_response(response: Any, method: str) -> Any:
    """Validate the RPC response envelope; raise ``RpcError``, else return result."""
    if not isinstance(response, dict):
        raise RpcError(method, f"bad response type: {type(response).__name__}")
    if not response.get("ok"):
        raise RpcError(
            method,
            str(response.get("error", "<no error message>")),
            traceback=response.get("traceback"),
        )
    return response.get("result")


def make_error_response(exc: BaseException) -> dict:
    """Build the error envelope for a caught exception (upstream rpc_facade)."""
    return {
        "ok": False,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


# ---------------------------------------------------------------------------
#  JSON <-> ndarray (vendor/rpent/utils/rpc/http_rpc.py)
# ---------------------------------------------------------------------------


def _from_json(obj: Any) -> Any:
    """Rehydrate tagged ndarrays / numpy scalars; pass everything else through."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj and set(obj) <= {"__ndarray__", "dtype", "shape"}:
            raw = base64.b64decode(obj["__ndarray__"])
            arr = np.frombuffer(raw, dtype=obj.get("dtype"))
            # frombuffer returns a read-only view of the base64 bytes; copy so
            # callers can mutate the returned array like a pickled round-trip.
            return arr.reshape(obj.get("shape", (-1,))).copy()
        if "__npscalar__" in obj and set(obj) <= {"__npscalar__", "dtype"}:
            return np.dtype(obj["dtype"]).type(obj["__npscalar__"])
        return {k: _from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json(v) for v in obj]
    return obj


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that tags numpy arrays and scalars for a faithful decode."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": base64.b64encode(
                    np.ascontiguousarray(obj).tobytes()
                ).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, np.generic):
            item = obj.item()
            if obj.dtype.kind in "biuf" and isinstance(item, (bool, int, float)):
                return {"__npscalar__": item, "dtype": str(obj.dtype)}
        return super().default(obj)


def _is_direct_url(url: str) -> bool:
    hostname = urllib.parse.urlsplit(url).hostname
    return hostname is not None and hostname.lower() in _DIRECT_HOSTS


class HttpRpcClient:
    """Client that talks to :class:`HttpRpcServer` via HTTP POST /call.

    ``127.0.0.1`` / ``localhost`` bypass the process proxy configuration (a pod
    with HTTP_PROXY set would otherwise route loopback traffic through it).
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if _is_direct_url(self._base_url)
            else None
        )

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        payload = {"method": method, "args": list(args), "kwargs": kwargs or {}}
        body = json.dumps(payload, cls=_NumpyEncoder).encode("utf-8")
        request_timeout = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S
        req = urllib.request.Request(
            f"{self._base_url}/call",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            open_request = self._opener.open if self._opener else urllib.request.urlopen
            with open_request(req, timeout=request_timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            # HTTPError is an OSError subclass; catch it first so the ok=False
            # body the server sent alongside the status code is still parsed.
            raw = exc.read()
        except OSError as exc:
            raise RpcError(method, f"HTTP request failed: {exc}") from exc
        try:
            response = _from_json(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise RpcError(method, f"invalid JSON response: {exc}") from exc
        return check_response(response, method)


class _HttpRpcHandler(BaseHTTPRequestHandler):
    """Handles POST /call with a JSON body, dispatching to ``server.dispatch``."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:
        if self.path != "/call":
            self.send_response(404)
            self.end_headers()
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            request = json.loads(body)
            method = request["method"]
            args = tuple(_from_json(v) for v in request.get("args", []))
            kwargs = {k: _from_json(v) for k, v in request.get("kwargs", {}).items()}
            result = self.server.dispatch(method, args, kwargs)  # type: ignore[attr-defined]
            response: dict = {"ok": True, "result": result}
        except Exception as exc:
            response = make_error_response(exc)
        # Always 200; failures are described inside the body via ok=False.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        try:
            self.wfile.write(json.dumps(response, cls=_NumpyEncoder).encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError) as exc:
            logger.debug("rpc http write failed: %s", exc)


class HttpRpcServer(ThreadingHTTPServer):
    """HTTP server that dispatches JSON-RPC calls."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self, server_address: tuple[str, int], dispatch: Callable[..., Any]
    ) -> None:
        super().__init__(server_address, _HttpRpcHandler)
        self.dispatch = dispatch


# ---------------------------------------------------------------------------
#  Server facade (vendor/rpent/utils/rpc/rpc_facade.py, sessionless)
# ---------------------------------------------------------------------------


class RpcFacade:
    """Base for the env service: owns the method table, healthz/shutdown, teardown.

    Subclasses register handlers in ``self._rpc`` from ``_register_rpc`` (called
    by ``__init__``).
    """

    def __init__(self) -> None:
        self._shutdown_event = threading.Event()
        self._rpc: dict[str, Callable] = {}
        self._register_rpc()

    def _register_rpc(self) -> None:
        """Hook: subclasses populate ``self._rpc`` here."""

    def close(self) -> None:
        """Release resources. Override in subclasses that hold any."""

    def _builtin_dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """Handle framework methods; return ``None`` for business methods."""
        if method == "healthz":
            return {"status": "ok"}
        if method == "shutdown":
            self._shutdown_event.set()
            return {"ok": True}
        return None

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        result = self._builtin_dispatch(method, args, kwargs)
        if result is not None:
            return result
        handler = self._rpc.get(method)
        if handler is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        return handler(*args, **kwargs)

    def _bind_and_announce(
        self,
        transport: Literal["http"],
        host: str,
        port: int,
        dispatch: Callable[..., Any],
    ) -> HttpRpcServer:
        """Bind the transport server and print the URL a client should dial.

        ``server_address`` reflects the actually-bound port, which is what makes
        ``--port 0`` usable.
        """
        if transport != "http":
            raise ValueError(f"only the http transport is vendored here, got {transport!r}")
        server = HttpRpcServer((host, port), dispatch)
        bound_host, bound_port = server.server_address[:2]
        client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        print(f"RPC server listening on http://{client_host}:{bound_port}", flush=True)
        logger.info("RPC server listening on http://%s:%s", client_host, bound_port)
        return server

    def serve(
        self,
        *,
        transport: Literal["http"] = "http",
        host: str = "127.0.0.1",
        port: int = 0,
        parent_watch: bool = False,
    ) -> None:
        """Bind, announce, serve until shutdown. See ``MainThreadServeMixin.serve``."""
        server = self._bind_and_announce(transport, host, port, self._dispatch)
        if parent_watch:
            watch_parent_death(self._shutdown_event.set)
        try:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self._shutdown_event.wait()
        finally:
            server.shutdown()
            server.server_close()
            self.close()


class MainThreadServeMixin:
    """Mixin overriding ``serve`` so every dispatch runs on ONE thread.

    Verbatim in behaviour from ``vendor/rpent/utils/rpc/main_thread_serve.py``.
    MuJoCo's EGL rendering context is thread-bound: every env op must run on the
    same thread that created it. The transport server therefore runs on a daemon
    thread and ``_dispatch_main_thread`` (the transport-side proxy) enqueues each
    request onto ``_main_thread_queue``, blocking until the consumer loop -- which
    runs on the thread that called ``serve`` -- has executed it. Mix in AHEAD of
    :class:`RpcFacade`; subclasses inherit ``serve`` as-is.
    """

    def _dispatch_main_thread(self, method: str, args: tuple, kwargs: dict) -> Any:
        result = self._builtin_dispatch(method, args, kwargs)  # type: ignore[attr-defined]
        if result is not None:
            if method == "shutdown":
                self._main_thread_queue.put(None)  # sentinel: wake the consumer
            return result
        event = threading.Event()
        req: dict = {
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "result": None,
            "error": None,
        }
        self._main_thread_queue.put((event, req))
        event.wait()
        if req["error"]:
            raise RuntimeError(req["error"])
        return req["result"]

    def serve(
        self,
        *,
        transport: Literal["http"] = "http",
        host: str = "127.0.0.1",
        port: int = 0,
        parent_watch: bool = False,
    ) -> None:
        """Serve RPC, dispatching every call on the calling (main) thread."""
        self._main_thread_queue: "queue.Queue[tuple[threading.Event, dict] | None]" = (
            queue.Queue()
        )
        server = self._bind_and_announce(  # type: ignore[attr-defined]
            transport, host, port, self._dispatch_main_thread
        )
        if parent_watch:
            watch_parent_death(self._shutdown_event.set)  # type: ignore[attr-defined]
        # Dispatch runs on THIS thread; poll the queue so shutdown via the event
        # (parent-watch) or the None sentinel (shutdown RPC) both unblock it.
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            while not self._shutdown_event.is_set():  # type: ignore[attr-defined]
                try:
                    item = self._main_thread_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                event, req = item
                try:
                    req["result"] = self._dispatch(  # type: ignore[attr-defined]
                        req["method"], req["args"], req["kwargs"]
                    )
                except Exception:
                    req["error"] = traceback.format_exc()
                event.set()
        finally:
            server.shutdown()
            server.server_close()
            self.close()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
#  Client helpers (vendor/rpent/utils/rpc/client_utils.py)
# ---------------------------------------------------------------------------


def parse_endpoint(endpoint: str) -> tuple[str, str, int]:
    """Parse ``[protocol://]host:port`` into ``(protocol, host, port)``."""
    if "://" in endpoint:
        protocol, _, rest = endpoint.partition("://")
    else:
        protocol, rest = "http", endpoint
    host, _, port = rest.partition(":")
    if not host or not port:
        raise ValueError(f"endpoint must be [protocol://]host:port, got {endpoint!r}")
    return protocol, host, int(port)


def make_rpc_client(endpoint: str) -> HttpRpcClient:
    protocol, host, port = parse_endpoint(endpoint)
    if protocol != "http":
        raise ValueError(f"endpoint protocol must be http, got {protocol!r}")
    return HttpRpcClient(f"http://{host}:{port}")


def wait_for_ready(
    client: HttpRpcClient, *, timeout_s: float = 900.0, poll_interval_s: float = 0.5
) -> None:
    """Poll ``healthz`` until it answers or ``timeout_s`` elapses.

    The default timeout is generous because the FIRST readiness of this service
    includes a full RoboCasa env construction (scene generation + MuJoCo compile),
    which is minutes, not seconds, on a cold pod.
    """
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            client.call("healthz", timeout_s=1.0)
            return
        except Exception as exc:
            last_err = exc
            time.sleep(poll_interval_s)
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f}s: {last_err}")


__all__ = [
    "HttpRpcClient",
    "HttpRpcServer",
    "MainThreadServeMixin",
    "RpcError",
    "RpcFacade",
    "configure_egl_device",
    "make_rpc_client",
    "parse_endpoint",
    "wait_for_ready",
    "watch_parent_death",
]
