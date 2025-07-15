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
    model_path = "inclusionAI/Ming-Lite-Omni"
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda", dtype=torch.bfloat16)

    processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)

    vision_path = "figures/cases"

    # qa
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "请详细介绍鹦鹉的生活习性。"}
            ],
        },
    ]
    generate(messages=messages, processor=processor, model=model)

    # image qa
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": os.path.join(vision_path, "flower.jpg")},
                {"type": "text", "text": "What kind of flower is this?"},
            ],
        },
    ]
    generate(messages=messages, processor=processor, model=model)

    # video qa
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "video", "video": os.path.join(vision_path, "yoga.mp4"), 'min_pixels': 100352, 'max_pixels': 602112, "sample": "uniform"},
                {"type": "text", "text": "What is the woman doing?"},
            ],
        },
    ]
    generate(messages=messages, processor=processor, model=model)

    # multi-turn chat
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "中国的首都是哪里？"},
            ],
        },
        {
            "role": "ASSISTANT",
            "content": [
                {"type": "text", "text": "北京"},
            ],
        },
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "它的占地面积是多少？有多少常住人口？"},
            ],
        },
    ]
    generate(messages=messages, processor=processor, model=model)

