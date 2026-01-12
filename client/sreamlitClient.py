#!/usr/bin/env python3
"""
Simple Streamlit Chatbot for vLLM OpenAI-compatible server.

Usage:
  streamlit run streamlit_vllm_chatbot.py

"""
import streamlit as st
import requests
import os
import json
import time
import socket
from urllib.parse import urlparse, urlunparse

def build_headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def normalize_base_url(base_url: str) -> str:
    """Normalize URL by collapsing duplicate slashes in the path."""
    if not base_url:
        return base_url
    parsed = urlparse(base_url)
    path = parsed.path
    while "//" in path:
        path = path.replace("//", "/")
    parsed = parsed._replace(path=path)
    return urlunparse(parsed)


def host_resolves(hostname: str) -> bool:
    try:
        socket.getaddrinfo(hostname, None)
        return True
    except Exception:
        return False


def default_base_url() -> str:
    env_url = os.getenv("VLLM_BASE_URL")
    if env_url:
        return env_url
    public_url = "https://bida-ml.oss.net.bd/ba_llm"
    if host_resolves("bida-ml.oss.net.bd"):
        return public_url
    return "http://localhost:7080/ba_llm"

def chat_once(base_url, model, messages, temperature, max_tokens, stream, api_key, timeout_s):
    base_url = normalize_base_url(base_url)
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    resp = requests.post(
        url,
        headers=build_headers(api_key),
        json=payload,
        stream=stream,
        timeout=timeout_s,
    )
    if resp.status_code >= 400:
        try:
            st.error(resp.text[:2000])
        except Exception:
            pass
        resp.raise_for_status()
    if not stream:
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    reply = ""
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
            yield delta
            reply += delta
    return reply

def resolve_default_model(base_url, api_key, timeout_s):
    base_url = normalize_base_url(base_url)
    url = base_url.rstrip("/") + "/v1/models"
    try:
        resp = requests.get(url, headers=build_headers(api_key), timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data") or []
        if models:
            model_id = models[0].get("id")
            if isinstance(model_id, str) and model_id:
                return model_id
    except Exception:
        return None
    return None

def count_tokens_estimate(text):
    """Rough token estimation: ~4 chars per token"""
    return len(text) // 4

def get_conversation_tokens(messages):
    """Estimate total tokens in conversation"""
    total = 0
    for msg in messages:
        total += count_tokens_estimate(msg.get("content", ""))
    return total

def main():
    st.set_page_config(
        page_title="vLLM Chat", 
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # ChatGPT-like styling
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .stChatInput {
            border-radius: 1.5rem;
        }
        div[data-testid="stStatusWidget"] {
            visibility: hidden;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        base_url = st.text_input(
            "Base URL",
            default_base_url(),
        )
        base_url = normalize_base_url(base_url)

        # Help users running this on a server that doesn't have DNS for the public name.
        parsed = urlparse(base_url)
        if parsed.scheme in ("http", "https") and parsed.hostname:
            if parsed.hostname == "bida-ml.oss.net.bd" and not host_resolves(parsed.hostname):
                st.warning(
                    "এই মেশিনে `bida-ml.oss.net.bd` DNS resolve হচ্ছে না। "
                    "তাই পাবলিক ডোমেইন দিয়ে call করলে কানেক্ট হবে না। "
                    "সমাধান: এই মেশিনে `/etc/hosts` এ `114.130.242.114 bida-ml.oss.net.bd` যোগ করুন, "
                    "অথবা Base URL হিসেবে `http://localhost:7080/ba_llm` ব্যবহার করুন।"
                )
        api_key = st.text_input("API Key (optional)", type="password")
        
        st.divider()
        
        temperature = st.slider("Temperature", 0.0, 2.0, 0.6, 0.05)
        max_tokens = st.number_input("Max tokens per response", min_value=100, max_value=4000, value=2000, step=100)
        timeout_s = st.number_input("Timeout (s)", min_value=10, max_value=600, value=120)
        stream = st.checkbox("Enable streaming", value=True)
        
        st.divider()
        
        system_prompt = st.text_area(
            "System prompt",
            "You are a helpful assistant. Answer in Bangla and English as appropriate.",
            height=100,
        )
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"] = [{"role": "system", "content": system_prompt}]
            st.session_state["token_count"] = 0
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": system_prompt}]
    if "model" not in st.session_state:
        model = resolve_default_model(base_url, api_key, timeout_s) or "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
        st.session_state["model"] = model
    if "token_count" not in st.session_state:
        st.session_state["token_count"] = 0

    # Header with token counter
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("💬 Chat")
    with col2:
        tokens_used = get_conversation_tokens(st.session_state["messages"])
        st.metric("Tokens", f"{tokens_used}")
    
    st.caption(f"🤖 Model: `{st.session_state['model']}`")
    st.divider()

    # Display chat messages
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Message vLLM Chat..."):
        # Check token limit before adding message
        current_tokens = get_conversation_tokens(st.session_state["messages"])
        prompt_tokens = count_tokens_estimate(prompt)
        
        # Model context window is 8192, leave buffer for response
        if current_tokens + prompt_tokens > 6000:
            st.warning("⚠️ Token limit approaching! Reducing context automatically...")
            # Keep system prompt + last few messages
            st.session_state["messages"] = [
                st.session_state["messages"][0],  # system prompt
                {"role": "assistant", "content": "Previous conversation was too long. Continuing with recent context..."},
            ] + st.session_state["messages"][-4:]  # Keep last 4 messages
        
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            if stream:
                response = ""
                msg_placeholder = st.empty()
                start_time = time.time()
                
                try:
                    for delta in chat_once(
                        base_url,
                        st.session_state["model"],
                        st.session_state["messages"],
                        temperature,
                        max_tokens,
                        True,
                        api_key,
                        timeout_s,
                    ):
                        response += delta
                        msg_placeholder.markdown(response + "▌")
                    
                    msg_placeholder.markdown(response)
                    elapsed = time.time() - start_time
                    
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    st.caption(f"⏱️ {elapsed:.2f}s")
                    
                except Exception as e:
                    error_msg = str(e)
                    if "too large" in error_msg or "maximum context length" in error_msg:
                        st.error("❌ Token limit exceeded! Reducing context...")
                        # Auto-reduce context and retry
                        st.session_state["messages"] = [
                            st.session_state["messages"][0],
                            {"role": "assistant", "content": "Context reduced due to token limit. Please resend your message."},
                        ] + st.session_state["messages"][-2:]
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {error_msg}")
            else:
                try:
                    start_time = time.time()
                    response = chat_once(
                        base_url,
                        st.session_state["model"],
                        st.session_state["messages"],
                        temperature,
                        max_tokens,
                        False,
                        api_key,
                        timeout_s,
                    )
                    elapsed = time.time() - start_time
                    
                    st.markdown(response)
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    st.caption(f"⏱️ {elapsed:.2f}s")
                    
                except Exception as e:
                    error_msg = str(e)
                    if "too large" in error_msg or "maximum context length" in error_msg:
                        st.error("❌ Token limit exceeded! Reducing context...")
                        st.session_state["messages"] = [
                            st.session_state["messages"][0],
                            {"role": "assistant", "content": "Context reduced due to token limit. Please resend your message."},
                        ] + st.session_state["messages"][-2:]
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {error_msg}")
        
        st.rerun()

if __name__ == "__main__":
    main()
