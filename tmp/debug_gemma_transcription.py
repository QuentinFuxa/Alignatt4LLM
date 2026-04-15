"""Minimal Gemma transcription smoke check — isolate whether the basic
prompt + audio path produces a correct transcription.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_alignment_single_audio import load_wav


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = "/home/fuxa/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"
    audio_path = "tmp/alignatt_smoke18.wav"

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    audio, sr = load_wav(audio_path)
    print(f"audio len={len(audio)}  sr={sr}  min={audio.min():.3f}  max={audio.max():.3f}  mean={audio.mean():.4f}")
    assert sr == 16000

    for prompt_variant, prompt_text in [
        ("cookbook_default",
         "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."),
        ("cookbook_english",
         "Transcribe the following speech segment in English into English text. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."),
    ]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "audio", "audio": audio},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Cast float tensors to model dtype.
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(model.dtype)

        audio_token_id = int(getattr(model.config, "audio_token_id"))
        input_ids_list = inputs["input_ids"][0].tolist()
        num_audio = sum(1 for t in input_ids_list if t == audio_token_id)
        print(f"\n=== variant: {prompt_variant}")
        print(f"prompt_len={len(input_ids_list)}  audio_tokens={num_audio}")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        text = processor.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(f"transcript: {text}")


if __name__ == "__main__":
    main()
