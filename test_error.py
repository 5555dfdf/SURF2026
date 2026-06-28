import json
from urllib import request

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"

prompt = """
You are an expert annotator.

Return ONLY a valid JSON object.

{
    "relevant":"0|1",
    "sentiment":"0|1|2|3",
    "confidence":0.0,
    "reason":"..."
}

Tweet:

The government should ban children under 16 from using social media.
"""

payload = {
    "model": "qwen3:4b",

    # Chat API
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ],

    # 非流式
    "stream": False,

    # 关闭 thinking
    "think": False,

    # 强制 JSON
    "format": {
        "type": "object",
        "properties": {
            "relevant": {
                "type": "string"
            },
            "sentiment": {
                "type": "string"
            },
            "confidence": {
                "type": "number"
            },
            "reason": {
                "type": "string"
            }
        },
        "required": [
            "relevant",
            "sentiment",
            "confidence",
            "reason"
        ]
    },

    "options": {
        "temperature": 0
    }
}

body = json.dumps(payload).encode("utf-8")

req = request.Request(
    OLLAMA_CHAT_URL,
    data=body,
    headers={
        "Content-Type": "application/json"
    },
    method="POST"
)

print("=" * 80)
print("Sending request...")
print("=" * 80)

with request.urlopen(req, timeout=120) as resp:

    print("HTTP Status:", resp.status)
    print()

    text = resp.read().decode("utf-8")

print("=" * 80)
print("Raw Response")
print("=" * 80)
print(text)

response = json.loads(text)

print()
print("=" * 80)
print("Pretty JSON")
print("=" * 80)
print(json.dumps(response, indent=4, ensure_ascii=False))

print()

if "message" in response:

    message = response["message"]

    print("=" * 80)
    print("Assistant Role")
    print("=" * 80)
    print(message.get("role"))

    print()

    print("=" * 80)
    print("Thinking")
    print("=" * 80)
    print(repr(message.get("thinking", "")))

    print()

    print("=" * 80)
    print("Content")
    print("=" * 80)
    print(message.get("content", ""))

else:
    print("No message field!")