import os
import sys
os.environ["VLLM_USE_V1"] = "0"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt as LLMInputs
from transformers import AutoTokenizer, AutoProcessor


class MingOmniImageQA(object):

    def __init__(self, bl_mm, temperature=0, max_tokens=512):
        self.bl_mm = bl_mm
        self.temperature = temperature
        self.max_tokens = max_tokens
        sys.path.insert(0, bl_mm)

        self.model_vllm = LLM(
            model=bl_mm,
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            tensor_parallel_size=1,
            limit_mm_per_prompt={"image": 10}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(bl_mm, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(bl_mm, trust_remote_code=True)

    def image_qa(self, messages_list):
        try:
            requests = []
            for messages in messages_list:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, use_system=True)
                image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)
                requests.append(LLMInputs({"prompt": text, "multi_modal_data": {"image": image_inputs}}))

            sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens)
            outputs = self.model_vllm.generate(requests, sampling_params)
            results = [output.outputs[0].text for output in outputs]
            return results
        except Exception as e:
            print(f"Error during vision QA processing: {e}")
            return []


if __name__ == "__main__":
    bl_mm = "/hetero_infer_new/serina.wzq/bailingv4_moe_lite/"
    model = MingOmniImageQA(bl_mm)

    vision_path = "https://img1.baidu.com/it/u=3997625245,1265562944&fm=253&app=138&f=JPEG?w=800&h=1428$0"
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": vision_path},
                {"type": "text", "text": "What kind of flower is this?"},
            ],
        },
    ]

    response = model.image_qa([messages])
    print("Generated Response:", response)
