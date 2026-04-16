"""Quick test to verify an OpenAI API key works for GPT-5-mini."""
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
    response = client.responses.create(
        model="gpt-5-mini",
        input="Say hello in one word.",
        max_output_tokens=16,
    )
    print("Token is valid!")
    print(f"Response: {response.output_text}")
except Exception as e:
    print(f"Token check failed: {e}")
