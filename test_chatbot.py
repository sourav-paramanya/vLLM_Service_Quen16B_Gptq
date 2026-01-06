import requests
import uuid

API_URL = "http://localhost:8080/v1/chat/completions"
API_KEY = "testkey123"  # 8+ char, any string
MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"


def chat(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": API_KEY
    }
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    print("Simple AI Chatbot (Llama 3.1, local API)")
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant."}
    ]
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        messages.append({"role": "user", "content": user_input})
        try:
            reply = chat(messages)
            print(f"Bot: {reply}")
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"[Error] {e}")
            break

if __name__ == "__main__":
    main()
