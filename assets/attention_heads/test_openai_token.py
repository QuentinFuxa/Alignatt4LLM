"""Quick test to verify an OpenAI API key works."""
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package not installed. Run: pip install openai")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    key_file = os.path.expanduser("~/.openai_api_key")
    if os.path.exists(key_file):
        api_key = open(key_file).read().strip()
    else:
        sys.exit("Set OPENAI_API_KEY env var or create ~/.openai_api_key")

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    print("Token is valid!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Token check failed: {e}")
