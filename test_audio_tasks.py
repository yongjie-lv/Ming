import os
import time
import torch
from transformers import AutoProcessor

from modeling_bailingmm import BailingMMNativeForConditionalGeneration

def generate(messages, processor, model):
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        audio_kwargs={'use_whisper_encoder': True}
    ).to(model.device)

    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    srt_time = time.time()
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=processor.gen_terminator,
    )
    generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(output_text)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")


if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
    model_path = "."
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")

    # ASR
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text",
                 "text": "Please recognize the language of this speech and transcribe it. Format: oral."},
                {"type": "audio", "audio": 'data/wavs/BAC009S0915W0292.wav'},
            ],
        },
    ]

    generate(messages=messages, processor=processor, model=model)
