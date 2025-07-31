import os
os.environ["VLLM_USE_V1"] = "0"


import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt as LLMInputs
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
import sys
import torch

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
ming_lite_path = "inclusionAI/Ming-lite-1.5"

sys.path.insert(0, ming_lite_path)

if __name__=="__main__":
    llm = LLM(model=ming_lite_path, trust_remote_code=True, enforce_eager=False, disable_custom_all_reduce=True, tensor_parallel_size=1, limit_mm_per_prompt={"image": 10})

    tokenizer = AutoTokenizer.from_pretrained(ming_lite_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(ming_lite_path, trust_remote_code=True)

    vision_path = "path/to/vision_path"
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": os.path.join(vision_path, "flowers.jpg")},
                {"type": "text", "text": "这个里面的是什么东西，什么颜色?"},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, use_system=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "video", "video": os.path.join(vision_path, "yoga.mp4"), 'min_pixels': 100352, 'max_pixels': 602112, "sample": "uniform"},
                {"type": "text", "text": "What is the woman doing?"},
            ],
        },
    ]

    text_1 = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, use_system=True)
    image_inputs_1, video_inputs_1, audio_inputs_1 = processor.process_vision_info(messages)

    messages = [
       {
           "role": "HUMAN",
           "content": [
               {"type": "text", "text": "Please recognize the language of this speech and transcribe it. Format: oral."},
               {"type": "audio", "audio": '/path/to/BAC009S0915W0292.wav'},
           ],
        },
   ]

    text_2 = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, use_system=True)

    image_inputs_2, video_inputs_2, audio_inputs_2 = processor.process_vision_info(messages)

    message_2 = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": "/path/to/0.jpg"},
                {"type": "image", "image": "/path/to/1.jpg"},
                {"type": "image", "image": os.path.join(vision_path, "flowers.jpg")},
                {"type": "text", "text": "Question: Which option describe the object relationship in the image correctly?\nOptions:\nA. The suitcase is on the book.\nB. The suitcase is beneath the cat.\nC. The suitcase is beneath the bed.\nD. The suitcase is beneath the book.\nPlease select the correct answer from the options above."},
            ],
        },
    ]
    text_3 = processor.apply_chat_template(message_2, tokenize=False, add_generation_prompt=True, use_system=True)
    image_inputs_3, video_inputs_3, audio_inputs_3 = processor.process_vision_info(message_2)



    requests = [LLMInputs({ "prompt": text, "multi_modal_data": {"image": image_inputs} }),
                LLMInputs({ "prompt": text, "multi_modal_data": {"image": image_inputs} }),
                LLMInputs({ "prompt": text_3, "multi_modal_data": {"image": image_inputs_3} }),
                LLMInputs({ "prompt": text_1, "multi_modal_data": {"video": video_inputs_1} }),
                LLMInputs({ "prompt": text_2, "multi_modal_data": {"audio": audio_inputs_2} }),
    ]
    print(requests)
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    outputs = llm.generate(requests, sampling_params)
    print(outputs)