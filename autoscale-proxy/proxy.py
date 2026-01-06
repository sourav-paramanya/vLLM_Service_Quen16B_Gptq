#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

import aiohttp
from aiohttp import web
import docker


def env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return int(val)
    except Exception:
        return default


LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = env_int("LISTEN_PORT", 8080)

# Internal URL reachable from this container (same docker network)
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://vllm-server:8080")

VLLM_CONTAINER_NAME = os.environ.get("VLLM_CONTAINER_NAME", "vllm-server")
IDLE_TIMEOUT_SECONDS = env_int("IDLE_TIMEOUT_SECONDS", 600)
CHECK_INTERVAL_SECONDS = env_int("CHECK_INTERVAL_SECONDS", 30)
WARMUP_TIMEOUT_SECONDS = env_int("WARMUP_TIMEOUT_SECONDS", 900)

LAST_REQUEST_FILE = os.environ.get("LAST_REQUEST_FILE", "/tmp/vllm_last_request")

# How long we wait for backend per request before giving up.
PER_REQUEST_WAIT_SECONDS = env_int("PER_REQUEST_WAIT_SECONDS", 300)

# Limit max request body to avoid memory blow-ups
MAX_BODY_BYTES = env_int("MAX_BODY_BYTES", 50 * 1024 * 1024)


class State:
    def __init__(self) -> None:
        self.active_requests = 0
        self._lock = asyncio.Lock()
        self.docker_client = docker.from_env()

    async def inc_active(self) -> None:
        async with self._lock:
            self.active_requests += 1

    async def dec_active(self) -> None:
        async with self._lock:
            self.active_requests = max(0, self.active_requests - 1)

    async def get_active(self) -> int:
        async with self._lock:
            return self.active_requests


STATE = State()


def touch_last_request() -> None:
    try:
        with open(LAST_REQUEST_FILE, "w", encoding="utf-8") as f:
            f.write(str(int(time.time())))
    except Exception:
        # best-effort
        pass


def read_last_request_ts() -> Optional[int]:
    try:
        with open(LAST_REQUEST_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def is_container_running() -> bool:
    try:
        c = STATE.docker_client.containers.get(VLLM_CONTAINER_NAME)
        c.reload()
        return c.status == "running"
    except Exception:
        return False


def start_container() -> None:
    c = STATE.docker_client.containers.get(VLLM_CONTAINER_NAME)
    c.start()


def stop_container() -> None:
    c = STATE.docker_client.containers.get(VLLM_CONTAINER_NAME)
    c.stop(timeout=30)


async def wait_for_backend(timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    url = BACKEND_BASE_URL.rstrip("/") + "/v1/models"

    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def filtered_request_headers(request: web.Request) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS:
            continue
        if lk == "host":
            continue
        out[k] = v
    return out


async def proxy_handler(request: web.Request) -> web.StreamResponse:
    await STATE.inc_active()
    try:
        touch_last_request()

        # Ensure backend is running
        if not is_container_running():
            try:
                start_container()
            except Exception as e:
                return web.json_response(
                    {"error": {"message": f"Failed to start vLLM container: {e}", "type": "StartupError"}},
                    status=503,
                )

        # Wait for backend to be ready
        ok = await wait_for_backend(min(PER_REQUEST_WAIT_SECONDS, WARMUP_TIMEOUT_SECONDS))
        if not ok:
            return web.json_response(
                {"error": {"message": "vLLM backend is starting; try again shortly.", "type": "WarmingUp"}},
                status=503,
            )

        # Build backend URL
        path_qs = request.rel_url.raw_path
        if request.rel_url.raw_query_string:
            path_qs += "?" + request.rel_url.raw_query_string
        backend_url = BACKEND_BASE_URL.rstrip("/") + path_qs

        body = await request.read()
        if len(body) > MAX_BODY_BYTES:
            return web.json_response(
                {"error": {"message": f"Request body too large (> {MAX_BODY_BYTES} bytes)", "type": "RequestTooLarge"}},
                status=413,
            )

        timeout = aiohttp.ClientTimeout(total=PER_REQUEST_WAIT_SECONDS)
        headers = filtered_request_headers(request)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method=request.method,
                url=backend_url,
                headers=headers,
                data=body if body else None,
                allow_redirects=False,
            ) as resp:
                # Stream response back to client (works for SSE streaming too)
                out = web.StreamResponse(status=resp.status)
                for k, v in resp.headers.items():
                    lk = k.lower()
                    if lk in HOP_BY_HOP_HEADERS:
                        continue
                    # Let aiohttp handle content-length for streaming
                    if lk == "content-length":
                        continue
                    out.headers[k] = v

                await out.prepare(request)

                async for chunk in resp.content.iter_chunked(64 * 1024):
                    await out.write(chunk)

                await out.write_eof()
                return out

    finally:
        await STATE.dec_active()


async def idle_monitor(app: web.Application) -> None:
    # On startup, initialize last_request to now (avoid immediate shutdown)
    if read_last_request_ts() is None:
        touch_last_request()

    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

            if not is_container_running():
                continue

            active = await STATE.get_active()
            if active > 0:
                continue

            last_ts = read_last_request_ts() or int(time.time())
            idle_for = int(time.time()) - last_ts
            if idle_for >= IDLE_TIMEOUT_SECONDS:
                try:
                    stop_container()
                except Exception:
                    pass
        except Exception:
            # never crash the proxy
            continue


async def on_startup(app: web.Application) -> None:
    app["idle_task"] = asyncio.create_task(idle_monitor(app))


async def on_cleanup(app: web.Application) -> None:
    task = app.get("idle_task")
    if task:
        task.cancel()
        with contextlib.suppress(Exception):
            await task


def create_app() -> web.Application:
    app = web.Application(client_max_size=MAX_BODY_BYTES)
    app.router.add_route("*", "/{tail:.*}", proxy_handler)
    app.on_startup.append(on_startup)
    return app


if __name__ == "__main__":
    import contextlib

    web.run_app(create_app(), host=LISTEN_HOST, port=LISTEN_PORT)
