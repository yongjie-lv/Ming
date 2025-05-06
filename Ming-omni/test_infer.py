import os
import torch
from transformers import AutoProcessor

from modeling_bailingmm import BailingMMNativeForConditionalGeneration


if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
    model_path = "MODEL_PATH"
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    ).to("cuda")

    # replace with your model path
    vision_path = environment.get("VISION_PATH", "") or os.path.join(os.path.dirname(__file__), "vision")

    # qa
    # messages = [
    #     {
    #         "role": "HUMAN",
    #         "content": [
    #             {"type": "text", "text": "请详细介绍鹦鹉的生活习性。"}
    #         ],
    #     },
    # ]

    # image qa
    # messages = [
    #     {
    #         "role": "HUMAN",
    #         "content": [
    #             {"type": "image", "image": os.path.join(vision_path, "flowers.jpg")},
    #             {"type": "text", "text": "What kind of flower is this?"},
    #         ],
    #     },
    # ]

    # # video qa
    # messages = [
    #     {
    #         "role": "HUMAN",
    #         "content": [
    #             {"type": "video", "video": os.path.join(vision_path, "yoga.mp4")},
    #             {"type": "text", "text": "What is the woman doing?"},
    #         ],
    #     },
    # ]

    # multi-turn chat
    # messages = [
    #     {
    #         "role": "HUMAN",
    #         "content": [
    #             {"type": "text", "text": "中国的首都是哪里？"},
    #         ],
    #     },
    #     {
    #         "role": "ASSISTANT",
    #         "content": [
    #             {"type": "text", "text": "北京"},
    #         ],
    #     },
    #     {
    #         "role": "HUMAN",
    #         "content": [
    #             {"type": "text", "text": "它的占地面积是多少？有多少常住人口？"},
    #         ],
    #     },
    # ]

    # notice place the audio file in the same directory as the output file
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "Please recognize the language of this speech and transcribe it. Format: oral."},
                {"type": "audio", "audio": os.path.join(vision_path, "audio.wav")},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=False,
        eos_token_id=processor.gen_terminator,
    )
    generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(output_text)
