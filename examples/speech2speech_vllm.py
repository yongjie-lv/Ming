import os
os.environ["VLLM_USE_V1"] = "0"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt as LLMInputs
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
import torch
import re
import time
from typing import Dict, Optional
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from audio_detokenizer.cli.frontend import TTSFrontEnd
from modeling_bailing_talker import AudioDetokenizer, BailingTalkerForConditionalGeneration


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def initialize_models(model_path: str, spk_info: Dict[str, torch.Tensor]):
    model = LLM(model=model_path, trust_remote_code=True, enforce_eager=True, disable_custom_all_reduce=True, tensor_parallel_size=1, limit_mm_per_prompt={"image": 10}, gpu_memory_utilization=0.6)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    with open(f"{model_path}/talker/audio_detokenizer_stream.yaml", "r") as f:
        configs = load_hyperpyyaml(f)

    audio_detokenizer = AudioDetokenizer(
        f"{model_path}/talker/audio_detokenizer_stream.yaml",
        flow_model_path=f"{model_path}/talker/flow_stream.pt",
        hifigan_model_path=f"{model_path}/talker/hift_v2.pt",
        spk_info=spk_info,
    )
    # new mel
    audio_frontend = TTSFrontEnd(
        configs["feat_extractor"],
        f"{model_path}/talker/campplus.onnx",
        f"{model_path}/talker/speech_tokenizer_v1.onnx",
    )

    try:
        use_fp16 = False
        trt_file_name = 'flow.decoder.estimator.fp16.plan' if use_fp16 else "flow.decoder.estimator.fp32.plan"
        flow_decoder_onnx_model = os.path.join(model_path, 'talker', 'flow.decoder.estimator.fp32.onnx')
        flow_decoder_trt_model = os.path.join(model_path, 'talker', trt_file_name)
        audio_detokenizer.model.load_trt(flow_decoder_trt_model, flow_decoder_onnx_model, fp16=use_fp16)
    except Exception as e:
        print(f"load tensorrt file failed: {e}")

    return model, tokenizer, processor, audio_detokenizer, audio_frontend

def generate_tts(
        tts_text: str,
        prompt_text: str,
        prompt_wav_path: str,
        audio_frontend: TTSFrontEnd,
        talker: BailingTalkerForConditionalGeneration,
        audio_detokenizer: AudioDetokenizer,
        output_audio_path: Optional[str] = None,
        stream: bool = False,
):
    spk_input = audio_frontend.frontend_zero_shot(prompt_text, prompt_wav_path)

    is_chinese = contains_chinese(tts_text)
    # support english
    if not is_chinese:
        tts_text = tts_text.split()

    all_wavs = []
    start_time = time.perf_counter()
    for tts_speech, _ in talker.omni_audio_generation(
            tts_text, audio_detokenizer=audio_detokenizer, stream=stream, **spk_input
    ):
        all_wavs.append(tts_speech)
    waveform = torch.cat(all_wavs, dim=-1)
    if output_audio_path:
        torchaudio.save(output_audio_path, waveform, audio_detokenizer.sr)
    end_time = time.perf_counter()
    print(f"inference time cost: {end_time - start_time}")

    return waveform

if __name__=="__main__":
    MODEL_PATH = '/hetero_infer_new/kelv.wz/bailingv4_moe_lite_FP8'
    spk_info = {
        'luna': torch.load('data/spks/luna_v2.pt'),
        'eng': torch.load('data/spks/eng_v2.pt'),
    }
    model, tokenizer, processor, audio_detokenizer, audio_frontend = initialize_models(MODEL_PATH, spk_info)
    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    messages = [
       {
           "role": "HUMAN",
           "content": [
               {"type": "audio", "audio": '/hetero_infer_new/serina.wzq/bailingv4_moe_lite/data/wavs/speechQA_sample.wav'},
           ],
        },
   ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, use_system=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    requests = [LLMInputs({"prompt": text, "multi_modal_data": {"audio": audio_inputs}})]

    outputs = model.generate(requests, sampling_params)
    text = outputs[0].outputs[0].text
    print(text)

    talker = BailingTalkerForConditionalGeneration.from_pretrained(f'{MODEL_PATH}/talker').to(torch.bfloat16)

    # TTS Generation
    tts_wave = generate_tts(
        tts_text=text,
        prompt_text="感谢你的认可。",
        prompt_wav_path="data/spks/prompt.wav",
        audio_frontend=audio_frontend,
        talker=talker,
        audio_detokenizer=audio_detokenizer,
        output_audio_path="out_tts.wav",
        stream=False,
    )
    print("TTS 音频响应已保存到 out_tts.wav")
