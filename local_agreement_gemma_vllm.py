# %%
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import torch

model_path = "google/gemma-4-E4B-it"
tensor_parallel_size = max(1, torch.cuda.device_count())

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
llm = LLM(
    model=model_path,
    tensor_parallel_size=tensor_parallel_size,
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    trust_remote_code=True
)

# %%
CONTENT = """
You are a professional translator from English to German.
The sentence is likely to be hardly interrupted in the middle. Please translate only what you are sure about, if you feel that you do not
have enough context to translate everything, please stop early.
Please translate the following to German :
hey bro what's up
"""

messages = [
    {"role": "user", "content": CONTENT}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = llm.generate(prompt, SamplingParams(temperature=0.0, max_tokens=1024))

print(outputs[0].outputs[0].text)


# %%
# HERE insert minimal qwen3-asr that uses vllm