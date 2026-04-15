"""Deep debug: Gemma 4 E4B free-run ASR hallucinates on our clip while
Google claims FLEURS WER ~0.08. Compare our setup against the cookbook
step by step: rendered prompt, dtypes, shapes, generate() vs manual loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_alignment_single_audio import load_wav


MODEL_PATH = "/home/fuxa/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    audio, sr = load_wav("tmp/alignatt_smoke18.wav")
    print(f"Our audio: len={len(audio)}  sr={sr}  min={audio.min():.3f}  max={audio.max():.3f}  mean={audio.mean():.4f}  rms={np.sqrt((audio**2).mean()):.4f}")

    # ------------------------------------------------------------------
    # (1) Verify the rendered chat template matches cookbook convention.
    # ------------------------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Transcribe the following speech segment in English into English text.\n\n"
                        "Follow these specific instructions for formatting the answer:\n"
                        "* Only output the transcription, with no newlines.\n"
                        "* When transcribing numbers, write the digits, i.e. write 1.7 "
                        "and not one point seven, and write 3 instead of three."
                    ),
                },
                {"type": "audio", "audio": audio},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    print("\n--- rendered chat template (first 400 chars) ---")
    print(rendered[:400])
    print("...")
    print("--- rendered chat template (last 200 chars) ---")
    print(rendered[-200:])

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    print("\n--- input keys and shapes/dtypes ---")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")

    input_ids = inputs["input_ids"][0]
    audio_token_id = int(getattr(model.config, "audio_token_id"))
    audio_count = int((input_ids == audio_token_id).sum().item())
    first_audio_idx = int((input_ids == audio_token_id).nonzero()[0, 0].item())
    print(f"\naudio_token_id={audio_token_id}  count={audio_count}  first_idx={first_audio_idx}")

    # ------------------------------------------------------------------
    # (2) Cast inputs exactly as cookbook: `.to(device, dtype)` on
    #     BatchFeature. That only upcasts floating tensors to model dtype
    #     and leaves ints alone.
    # ------------------------------------------------------------------
    inputs_cookbook = inputs.to(model.device, dtype=model.dtype)

    # ------------------------------------------------------------------
    # (3) Use model.generate() with default config (what cookbook does),
    #     not my manual greedy loop.
    # ------------------------------------------------------------------
    with torch.no_grad():
        out = model.generate(
            **inputs_cookbook,
            max_new_tokens=128,
            do_sample=False,
        )
    text_cookbook = processor.tokenizer.decode(
        out[0, inputs_cookbook["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print("\n--- cookbook-style generate() output ---")
    print(repr(text_cookbook))

    # ------------------------------------------------------------------
    # (4) Sanity: also decode WITH special tokens to see end-of-turn etc.
    # ------------------------------------------------------------------
    text_with_specials = processor.tokenizer.decode(
        out[0, inputs_cookbook["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )
    print("\n--- output with special tokens ---")
    print(repr(text_with_specials))

    # ------------------------------------------------------------------
    # (5) Now amplify the audio 2x and try again — sometimes low RMS
    #     pushes the model off distribution.
    # ------------------------------------------------------------------
    loud = np.clip(audio * 2.5, -1.0, 1.0).astype(np.float32)
    print(f"\nLoud audio: min={loud.min():.3f}  max={loud.max():.3f}  rms={np.sqrt((loud**2).mean()):.4f}")
    loud_messages = [dict(messages[0])]
    loud_messages[0]["content"] = [
        {"type": "text", "text": messages[0]["content"][0]["text"]},
        {"type": "audio", "audio": loud},
    ]
    loud_inputs = processor.apply_chat_template(
        loud_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)
    with torch.no_grad():
        out_loud = model.generate(**loud_inputs, max_new_tokens=128, do_sample=False)
    text_loud = processor.tokenizer.decode(
        out_loud[0, loud_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print("\n--- amplified audio output ---")
    print(repr(text_loud))

    # ------------------------------------------------------------------
    # (6) Try the cookbook's "in its original language" prompt.
    # ------------------------------------------------------------------
    cookbook_prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Transcribe the following speech segment in its original language. "
                        "Follow these specific instructions for formatting the answer:\n"
                        "* Only output the transcription, with no newlines.\n"
                        "* When transcribing numbers, write the digits, i.e. write 1.7 "
                        "and not one point seven, and write 3 instead of three."
                    ),
                },
                {"type": "audio", "audio": audio},
            ],
        }
    ]
    cb_inputs = processor.apply_chat_template(
        cookbook_prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)
    with torch.no_grad():
        out_cb = model.generate(**cb_inputs, max_new_tokens=128, do_sample=False)
    text_cb = processor.tokenizer.decode(
        out_cb[0, cb_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print("\n--- cookbook 'original language' prompt output ---")
    print(repr(text_cb))


if __name__ == "__main__":
    main()
