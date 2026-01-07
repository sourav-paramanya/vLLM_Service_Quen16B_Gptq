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

def build_headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def chat_once(base_url, model, messages, temperature, max_tokens, stream, api_key, timeout_s):
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

def main():
    st.set_page_config(page_title="vLLM Streamlit Chatbot", layout="wide")
    st.title("🤖 vLLM Streamlit Chatbot")
    st.caption("Simple Streamlit UI for vLLM OpenAI-compatible server")

    base_url = st.sidebar.text_input("Base URL", "http://localhost:8080")
    api_key = st.sidebar.text_input("API Key (optional)", type="password")
    timeout_s = st.sidebar.number_input("Timeout (s)", min_value=10, max_value=600, value=120)
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.6, 0.05)
    max_tokens = st.sidebar.number_input("Max tokens", min_value=16, max_value=8192, value=8192)
    system_prompt = st.sidebar.text_area(
        "System prompt",
        "You are a helpful assistant. Answer in Bangla and English as appropriate.",
        height=80,
    )
    stream = st.sidebar.checkbox("Enable streaming output", value=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": system_prompt}
        ]
    if "model" not in st.session_state:
        model = resolve_default_model(base_url, api_key, timeout_s) or "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
        st.session_state["model"] = model

    st.sidebar.write(f"**Model:** {st.session_state['model']}")

    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])

    if prompt := st.chat_input("Type your message and press Enter..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            if stream:
                response = ""
                msg_placeholder = st.empty()
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
                    msg_placeholder.markdown(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})
            else:
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
                st.write(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})

    if st.sidebar.button("Reset chat"):
        st.session_state["messages"] = [
            {"role": "system", "content": system_prompt}
        ]
        st.experimental_rerun()

if __name__ == "__main__":
    main()
