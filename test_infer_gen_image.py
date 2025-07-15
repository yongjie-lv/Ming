import os
import time
import torch
from transformers import AutoProcessor

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from IPython import embed
import torchvision
from PIL import Image
import re

import torch.nn as nn
from collections import defaultdict

def get_model_memory_usage(model: nn.Module):
    """
    计算 PyTorch 模型参数在显存中的占用情况，按精度分类。

    Args:
        model (nn.Module): 要分析的 PyTorch 模型。

    Returns:
        dict: 包含不同精度参数数量和显存占用（GB）的字典。
              例如：
              {
                  'float32': {'count': 1000, 'memory_gb': 0.004},
                  'float16': {'count': 500, 'memory_gb': 0.001},
                  'bfloat16': {'count': 200, 'memory_gb': 0.0004},
                  'other': {'count': 50, 'memory_gb': 0.0001}
              }
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Input must be a torch.nn.Module instance.")

    # 尝试将模型移动到 GPU，如果可用的话
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 存储不同精度参数的总数量和总字节数
    param_stats = defaultdict(lambda: {'count': 0, 'memory_bytes': 0})

    print(f"分析模型参数显存占用 (Device: {device})...")
    print("-" * 50)

    for name, param in model.named_parameters():
        param_dtype = str(param.dtype).split('.')[-1] # 获取 'float32', 'float16' 等
        num_elements = param.numel() # 参数中的元素总数
        element_byte_size = param.element_size() # 每个元素占用的字节数

        memory_bytes = num_elements * element_byte_size

        if param_dtype == 'float32':
            param_stats['float32']['count'] += num_elements
            param_stats['float32']['memory_bytes'] += memory_bytes
        elif param_dtype == 'float16':
            param_stats['float16']['count'] += num_elements
            param_stats['float16']['memory_bytes'] += memory_bytes
        elif param_dtype == 'bfloat16':
            param_stats['bfloat16']['count'] += num_elements
            param_stats['bfloat16']['memory_bytes'] += memory_bytes
        else:
            param_stats['other']['count'] += num_elements
            param_stats['other']['memory_bytes'] += memory_bytes
        
        # 可选：打印每个参数的详细信息
        # print(f"  - {name}: Shape={list(param.shape)}, Dtype={param_dtype}, Size={memory_bytes / (1024**2):.4f} MB")

    print("-" * 50)
    print("显存占用汇总：")
    
    results = {}
    total_memory_gb = 0
    total_param_count = 0

    for dtype, stats in param_stats.items():
        memory_gb = stats['memory_bytes'] / (1024**3) # 转换为 GB
        print(f"  {dtype.upper()}:")
        print(f"    参数数量: {stats['count']:,} 个")
        print(f"    显存占用: {memory_gb:.4f} GB")
        
        results[dtype] = {'count': stats['count'], 'memory_gb': memory_gb}
        total_memory_gb += memory_gb
        total_param_count += stats['count']

    print("-" * 50)
    print(f"总参数数量: {total_param_count:,} 个")
    print(f"总显存占用 (仅参数): {total_memory_gb:.4f} GB")
    print("-" * 50)

    return results


def get_closest_ratio(height: float, width: float, aspect_ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(aspect_ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return aspect_ratios[closest_ratio], float(closest_ratio)

def auto_translate(model, processor, text):
    if re.search(r'[\u4e00-\u9fff]', text):
        prefix = "Translate the Chinese phrase below into natural English. Return only the translation result without any explanations, prefixes, or formatting. Phrase to translate:"
        text = f"{prefix}{text}"
        messages = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
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
        
        #srt_time = time.time()
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=False,
            eos_token_id=processor.gen_terminator,
        )
        generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return text

def process_ratio(ori_h, ori_w):
    ASPECT_RATIO_512 = {
    "0.25": [256, 1024],
    "0.26": [256, 992],
    "0.27": [256, 960],
    "0.28": [256, 928],
    "0.32": [288, 896],
    "0.33": [288, 864],
    "0.35": [288, 832],
    "0.4": [320, 800],
    "0.42": [320, 768],
    "0.48": [352, 736],
    "0.5": [352, 704],
    "0.52": [352, 672],
    "0.57": [384, 672],
    "0.6": [384, 640],
    "0.68": [416, 608],
    "0.72": [416, 576],
    "0.78": [448, 576],
    "0.82": [448, 544],
    "0.88": [480, 544],
    "0.94": [480, 512],
    "1.0": [512, 512],
    "1.07": [512, 480],
    "1.13": [544, 480],
    "1.21": [544, 448],
    "1.29": [576, 448],
    "1.38": [576, 416],
    "1.46": [608, 416],
    "1.67": [640, 384],
    "1.75": [672, 384],
    "2.0": [704, 352],
    "2.09": [736, 352],
    "2.4": [768, 320],
    "2.5": [800, 320],
    "2.89": [832, 288],
    "3.0": [864, 288],
    "3.11": [896, 288],
    "3.62": [928, 256],
    "3.75": [960, 256],
    "3.88": [992, 256],
    "4.0": [1024, 256],
}

    #ori_h, ori_w = pil_img.size[1], pil_img.size[0]
    closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, aspect_ratios=ASPECT_RATIO_512)
    closest_size = list(map(lambda x: int(x), closest_size))
    closest_ratio = closest_ratio
    if closest_size[0] / ori_h > closest_size[1] / ori_w:
        resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
    else:
        resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]

    return closest_size, resize_size #, closest_ratio

def generate_gen_image(
    model,
    processor,
    prompt,
    height=None,
    width=None,
    input_image_path=None,
    use_auto_translate=True,
    debug=False,
    POSITIVE_PREFIX_T2I=None,
    POSITIVE_PREFIX_I2I="high quality",
    NEGATIVE_PREFIX="worst quality, low quality, bad eyes, bad iris, twisted face, blurry, bad hand, watermark, multiple limbs, deformed fingers, bad fingers, ugly, monochrome, horror, geometry, bad anatomy, bad limbs, Blurry pupil, bad shading, error, bad composition, Extra fingers, strange fingers, Extra ears, extra leg, bad leg, disability, Blurry eyes, bad eyes, Twisted body, confusion, bad legs",
    image_gen_steps=30,
):
    # input_image_path 设置为 None, 运行 文生图，否则 图生图
    if height is None or width is None:
        image_gen_width, image_gen_height = 512 * 1, 512 * 1
    else:
        image_gen_width, image_gen_height = width, height

    closest_size, _ = process_ratio(ori_h=image_gen_height, ori_w=image_gen_width)
    image_gen_height, image_gen_width = closest_size[0] * 1, closest_size[1] * 1

    pil_img_tensor = None
    if input_image_path is not None:
        pil_img = Image.open(input_image_path)
        closest_size, resize_size = process_ratio(ori_h=pil_img.size[1], ori_w=pil_img.size[0])
        pil_img = torchvision.transforms.functional.resize(pil_img, resize_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        pil_img = torchvision.transforms.functional.center_crop(pil_img, closest_size)
        pil_img_tensor = (torchvision.transforms.functional.to_tensor(pil_img) - 0.5) * 2.0
        pil_img_tensor = pil_img_tensor.unsqueeze(0)
        print("ref pil_img_tensor shape: {}".format(pil_img_tensor.shape))
        if height is None or width is None:
            image_gen_width, image_gen_height = pil_img.size[0] * 1, pil_img.size[1] * 1

        if debug:
            pil_img.save("resized.jpg")
    
    if use_auto_translate:
        prompt_ori = prompt
        prompt = auto_translate(model, processor, prompt)
        if debug:
            print("prompt: {} -> translated: {}".format(prompt_ori, prompt))
    
    if input_image_path is not None:
        if POSITIVE_PREFIX_I2I:
            prompt = "{}; Requirement: {}".format(prompt, POSITIVE_PREFIX_I2I)
    else:
        if POSITIVE_PREFIX_T2I:
            prompt = "{}; Requirement: {}".format(prompt, POSITIVE_PREFIX_T2I)
    
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": prompt},
            ] if input_image_path is None else [
                {"type": "image", "image": input_image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if debug:
        print("messages:", messages)
    

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    #embed()

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

    if NEGATIVE_PREFIX:
        negative_messages = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": NEGATIVE_PREFIX},
                ] if input_image_path is None else [
                    {"type": "image", "image": input_image_path},
                    {"type": "text", "text": NEGATIVE_PREFIX},
                ],
            }
        ]
        if debug:
            print("negative_messages:", negative_messages)

        negative_text = processor.apply_chat_template(negative_messages, add_generation_prompt=True)
        negative_inputs = processor(
            text=[negative_text],
            images=image_inputs,
            videos=None,
            audios=None,
            return_tensors="pt",
        ).to(model.device)
        inputs["image_gen_negative_input_ids"] = negative_inputs["input_ids"]
        inputs["image_gen_negative_attention_mask"] = negative_inputs["attention_mask"]

    image = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=False,
        eos_token_id=processor.gen_terminator,
        image_gen=True,
        image_gen_height=image_gen_height,
        image_gen_width=image_gen_width,
        image_gen_steps=image_gen_steps,
        pixel_values_reference=pil_img_tensor,
    )
    
    return image


if __name__ == '__main__':
    
    model_path = "inclusionAI/Ming-Lite-Omni"
    processor = AutoProcessor.from_pretrained('.', trust_remote_code=True)
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        load_image_gen=True,
    ).to("cuda")
    
    image = generate_gen_image(
        model=model, 
        processor=processor,
        prompt="a beautiful girl wearing a red dress.",
        POSITIVE_PREFIX_T2I="",
        POSITIVE_PREFIX_I2I="",
        image_gen_steps=30,
        NEGATIVE_PREFIX="",
    )
    image.save("./woman_red.jpg")

    image = generate_gen_image(
        model=model, 
        processor=processor,
        prompt="给人物戴上墨镜",
        input_image_path="./woman_red.jpg",
        POSITIVE_PREFIX_T2I="",
        POSITIVE_PREFIX_I2I="",
        image_gen_steps=30,
        NEGATIVE_PREFIX="",
        use_auto_translate=False,
    )  
    image.save("./woman_red_sunglasses.jpg")

    
    
    