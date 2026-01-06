#!/usr/bin/env python3
"""Simple CLI chatbot for a local vLLM OpenAI-compatible server.

Default target:
  http://localhost:8080/v1/chat/completions

Usage examples:
  python3 test_vllm_chatbot.py
  python3 test_vllm_chatbot.py --system "তুমি একজন চাষি..." --stream
  python3 test_vllm_chatbot.py --base-url http://localhost:8080 --model hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4

Notes:
- vLLM does not require an API key by default. If you configured one behind a proxy,
  set VLLM_API_KEY and it will be sent as Authorization: Bearer <key>.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Optional


def _import_requests():
    try:
        import requests  # type: ignore

        return requests
    except Exception:
        print(
            "requests লাইব্রেরি পাওয়া যায়নি. ইনস্টল করুন: pip install requests\n",
            file=sys.stderr,
        )
        raise


def build_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def chat_once(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool,
    api_key: Optional[str],
    timeout_s: int,
) -> str:
    requests = _import_requests()

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    started = time.time()
    resp = requests.post(
        url,
        headers=build_headers(api_key),
        json=payload,
        stream=stream,
        timeout=timeout_s,
    )

    # helpful error body if non-200
    if resp.status_code >= 400:
        try:
            print(resp.text[:2000], file=sys.stderr)
        except Exception:
            pass
        resp.raise_for_status()

    if not stream:
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        elapsed = time.time() - started
        print(f"[latency] {elapsed:.2f}s", file=sys.stderr)
        return text

    # Streaming (SSE-like): lines begin with "data: {json}" and end with "data: [DONE]"
    reply = ""
    first_token_at: Optional[float] = None

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[len("data:") :].strip()
        if chunk == "[DONE]":
            break
        try:
            data = json.loads(chunk)
            delta = data["choices"][0].get("delta", {}).get("content", "")
        except Exception:
            continue

        if delta:
            if first_token_at is None:
                first_token_at = time.time()
                print(f"\n[ttft] {first_token_at - started:.2f}s", file=sys.stderr)
                print("Bot:", end=" ", flush=True)
            print(delta, end="", flush=True)
            reply += delta

    elapsed = time.time() - started
    if first_token_at is None:
        print("Bot:", end=" ", flush=True)
    print("\n", end="", flush=True)
    print(f"[latency] {elapsed:.2f}s", file=sys.stderr)
    return reply


def main() -> int:
    parser = argparse.ArgumentParser(description="CLI chatbot for local vLLM OpenAI API")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL of vLLM server (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--model",
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        help="Model id to request",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant. Answer in Bangla and English as appropriate.",
        help="System prompt",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output (recommended)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")

    args = parser.parse_args()

    stream = True
    if args.no_stream:
        stream = False
    elif args.stream:
        stream = True

    api_key = None
    # Optional: user can set env var if behind a proxy requiring auth
    try:
        import os

        api_key = os.environ.get("VLLM_API_KEY")
    except Exception:
        api_key = None

    messages: List[Dict[str, str]] = [{"role": "system", "content": args.system}]

    print("Simple vLLM Chatbot (type 'exit' to quit)")
    print(f"Target: {args.base_url}")
    print(f"Model:  {args.model}")
    print(f"Stream: {stream}")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            return 0

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Bye")
            return 0

        messages.append({"role": "user", "content": user_text})

        try:
            assistant_text = chat_once(
                base_url=args.base_url,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=stream,
                api_key=api_key,
                timeout_s=args.timeout,
            )
        except Exception as e:
            print(f"[Error] {e}", file=sys.stderr)
            return 1

        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    raise SystemExit(main())
