import os
import sys
os.environ["VLLM_USE_V1"] = "0"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt as LLMInputs
from transformers import AutoTokenizer, AutoProcessor
import torchaudio
import torch
import re
import time
from hyperpyyaml import load_hyperpyyaml
from audio_detokenizer.cli.frontend import TTSFrontEnd
from modeling_bailing_talker import AudioDetokenizer, BailingTalkerForConditionalGeneration
from typing import Dict, Optional
from PIL import Image

class MingOmni(object):

    def __init__(self, bl_mm, temperature=0, max_tokens=512):
        self.bl_mm = bl_mm
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.model_vllm = LLM(
            model=bl_mm,
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            tensor_parallel_size=1,
            limit_mm_per_prompt={"image": 10},
            gpu_memory_utilization=0.6,  # 0.6 for GPU with 48GB memory, 0.37 for GPU with 80GB memory
        )
        self.talker, self.audio_detokenizer, self.audio_frontend = self.init_talker()
        self.tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
        self.sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens)
        self.model_image_gen = self.init_image_gen()
        
    def init_image_gen(self):
        os.environ["IMAGE_GEN_MODE"] = "None"
        from modeling_bailingmm import BailingMMNativeForConditionalGeneration
        model_diffusion = BailingMMNativeForConditionalGeneration.from_pretrained(
            self.bl_mm,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
            attn_implementation="flash_attention_2",
            load_image_gen=True,
            low_cpu_mem_usage=True,       # Minimize CPU memory during loading
            load_vlm=False,               # No vlm, only diffusion
        ).to("cuda").to(torch.bfloat16)                     # Run on GPU

        return model_diffusion

    def init_talker(self):
        with open(f"{self.bl_mm}/talker/audio_detokenizer_stream.yaml", "r") as f:
            configs = load_hyperpyyaml(f)

        spk_info = {
            'luna': torch.load('data/spks/luna_v2.pt'),
            'eng': torch.load('data/spks/eng_v2.pt'),
        }

        audio_detokenizer = AudioDetokenizer(
            f"{self.bl_mm}/talker/audio_detokenizer_stream.yaml",
            flow_model_path=f"{self.bl_mm}/talker/flow_stream.pt",
            hifigan_model_path=f"{self.bl_mm}/talker/hift_v2.pt",
            spk_info=spk_info,
        )
        # new mel
        audio_frontend = TTSFrontEnd(
            configs["feat_extractor"],
            f"{self.bl_mm}/talker/campplus.onnx",
            f"{self.bl_mm}/talker/speech_tokenizer_v1.onnx",
        )

        try:
            use_fp16 = False
            trt_file_name = 'flow.decoder.estimator.fp16.plan' if use_fp16 else "flow.decoder.estimator.fp32.plan"
            flow_decoder_onnx_model = os.path.join(self.bl_mm, 'talker', 'flow.decoder.estimator.fp32.onnx')
            flow_decoder_trt_model = os.path.join(self.bl_mm, 'talker', trt_file_name)
            audio_detokenizer.model.load_trt(flow_decoder_trt_model, flow_decoder_onnx_model, fp16=use_fp16)
        except Exception as e:
            print(f"load tensorrt file failed: {e}")
        
        talker = BailingTalkerForConditionalGeneration.from_pretrained(f'{self.bl_mm}/talker').to(torch.bfloat16)

        return talker, audio_detokenizer, audio_frontend

    def qa(self, messages):
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, use_system=True)
            image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)

            if image_inputs is not None:
                requests = [LLMInputs({"prompt": text, "multi_modal_data": {"image": image_inputs}})]
            elif audio_inputs is not None:
                requests = [LLMInputs({"prompt": text, "multi_modal_data": {"audio": audio_inputs}})]
            elif video_inputs is not None:
                requests = [LLMInputs({"prompt": text, "multi_modal_data": {"video": video_inputs}})]
            else:
                requests = [LLMInputs({"prompt": text})]
            
            outputs = self.model_vllm.generate(requests, self.sampling_params)
            text = outputs[0].outputs[0].text
        except Exception as e:
            print(f"Error during vision QA processing: {e}")
            return None

        return text

    def generate_tts(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_wav_path: str,
        output_audio_path: Optional[str] = None,
        stream: bool = False,
    ):
        spk_input = self.audio_frontend.frontend_zero_shot(prompt_text, prompt_wav_path)

        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', tts_text))
        # support english
        if not is_chinese:
            tts_text = tts_text.split()

        all_wavs = []
        start_time = time.perf_counter()
        for tts_speech, _ in self.talker.omni_audio_generation(
                tts_text, audio_detokenizer=self.audio_detokenizer, stream=stream, **spk_input
        ):
            all_wavs.append(tts_speech)
        waveform = torch.cat(all_wavs, dim=-1)
        if output_audio_path:
            torchaudio.save(output_audio_path, waveform, self.audio_detokenizer.sr)
        end_time = time.perf_counter()
        print(f"inference time cost: {end_time - start_time}")

        return waveform

    def image_gen(self, prompt, image_path=None):
        assert len(prompt) > 0
        messages = [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": Image.new("RGB", (1,1), (0,0,0))},
                    ],
                },
        ] if image_path is None else [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, use_system=True)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)
        requests = [LLMInputs({ "prompt": text, "multi_modal_data": {"image": image_inputs} }),]
        sampling_params = SamplingParams(temperature=0, max_tokens=1, return_hidden_states=True)
        os.environ["IMAGE_GEN_MODE"] = "T2I" if image_path is None else "EDIT"
        outputs = self.model_vllm.generate(requests, sampling_params)
        os.environ["IMAGE_GEN_MODE"] = "None"
        prefill_hidden_states = outputs[0].prefill_hidden_states

        messages = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ] if image_path is None else [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                },
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)


        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        ).to(self.model_image_gen.device)

        for k in inputs.keys():
            if k in ["pixel_values", "pixel_values_videos", "audio_feats", "pixel_values_reference"]:
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        image = self.model_image_gen.generate(
            **inputs,
            image_gen_llm_hidden_states=prefill_hidden_states.unsqueeze(0),
            image_gen=True,
        )
        return image


if __name__ == "__main__":

    bl_mm = "YOUR_MODEL_PATH"
    model = MingOmni(bl_mm)

    # text qa
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "请详细介绍鹦鹉的生活习性。"}
            ],
        },
    ]
    response = model.qa(messages)
    print("Generated Response:", response)

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
    response = model.qa(messages)
    print("Generated Response:", response)

    # image qa
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
    response = model.qa(messages)
    print("Generated Response:", response)

    # image generate

    image_t2i = model.image_gen("a beautiful young women with red dress, standing on the beach")
    image_t2i.save("auto_t2i.jpg")
    print("Generated image: auto_t2i.jpg")

    image_edit = model.image_gen("给人物戴上墨镜", image_t2i)
    image_edit.save("auto_edit.jpg")
    print("Generated image: auto_edit.jpg")
    
    # speech qa
    messages = [
        {
           "role": "HUMAN",
           "content": [
               {"type": "audio", "audio": 'data/wavs/speechQA_sample.wav'},
           ],
        },
    ]
    response = model.qa(messages)
    print("Generated Response:", response)

    # tts
    tts_wave = model.generate_tts(
        tts_text=response,
        prompt_text="感谢你的认可。",
        prompt_wav_path="data/spks/prompt.wav",
        output_audio_path="out_tts.wav",
        stream=False,
    )
    print("Generated speech: out_tts.wav")
