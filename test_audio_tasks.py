import time
import warnings
from typing import Any, Dict, Optional

import re
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from transformers import AutoProcessor, GenerationConfig

from audio_detokenizer.cli.frontend import TTSFrontEnd
from modeling_bailing_talker import AudioDetokenizer
from modeling_bailingmm import BailingMMNativeForConditionalGeneration


warnings.filterwarnings("ignore")

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def initialize_models(model_path: str, spk_info: Dict[str, torch.Tensor], device: str = "cuda"):
    processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
    model = (
        BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to(device)
    )

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

    return processor, model, audio_detokenizer, audio_frontend


def generate(messages, processor: AutoProcessor, model: BailingMMNativeForConditionalGeneration):
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        audio_kwargs={"use_whisper_encoder": True},
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
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(output_text)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")
    return output_text


def generate_e2e(
    messages,
    processor: AutoProcessor,
    model: BailingMMNativeForConditionalGeneration,
    audio_detokenizer: AudioDetokenizer,
    spk_info: Dict[str, torch.Tensor],
    max_new_tokens: int = 512,
    speaker: str = "luna",
    output_audio_path: Optional[str] = None,
    use_whisper_encoder: bool = True,
    device: str = "cuda",
    generation_config: Optional[Dict[str, Any]] = None,
    stream: bool = False,
):

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, use_system=True
    )

    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        audio_kwargs={"use_whisper_encoder": use_whisper_encoder},
    ).to(device)

    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    if generation_config is None:
        generation_config = {"num_beams": 1}

    generation_config = GenerationConfig.from_dict(generation_config)

    srt_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=processor.gen_terminator,
            generation_config=generation_config,
            use_whisper_encoder=use_whisper_encoder,
        )

    generated_ids = outputs.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    is_chinese = contains_chinese(output_text)
    output_text = output_text.replace("-", "")
    # support english
    if not is_chinese:
        output_text = output_text.split()
    
    waveform = None
    if model.talker is not None and output_audio_path:
        thinker_reply_part = outputs.hidden_states[0][0] + outputs.hidden_states[0][-1]
        speaker = 'luna' if is_chinese else 'eng'
        spk_input = spk_info.get(speaker, "luna")

        all_wavs = []
        for tts_speech, text_list in model.talker.omni_audio_generation(
            output_text, audio_detokenizer=audio_detokenizer, thinker_reply_part=thinker_reply_part, speaker=speaker, stream=stream, **spk_input
        ):
            all_wavs.append(tts_speech)
        waveform = torch.cat(all_wavs, dim=-1)
        torchaudio.save(output_audio_path, waveform, audio_detokenizer.sr)

    print(f"Generate time: {(time.time() - srt_time):.2f}s")
    return output_text, waveform


def generate_tts(
    tts_text: str,
    prompt_text: str,
    prompt_wav_path: str,
    audio_frontend: TTSFrontEnd,
    model: BailingMMNativeForConditionalGeneration,
    audio_detokenizer: AudioDetokenizer,
    output_audio_path: Optional[str] = None,
    stream: bool = False,
):
    srt_time = time.time()
    spk_input = audio_frontend.frontend_zero_shot(prompt_text, prompt_wav_path)

    is_chinese = contains_chinese(tts_text)
    # support english
    if not is_chinese:
        tts_text = tts_text.split()

    all_wavs = []
    for tts_speech, text_list in model.talker.omni_audio_generation(
        tts_text, audio_detokenizer=audio_detokenizer, stream=stream, **spk_input
    ):
        all_wavs.append(tts_speech)
    waveform = torch.cat(all_wavs, dim=-1)
    torchaudio.save(output_audio_path, waveform, audio_detokenizer.sr)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")

    return waveform


if __name__ == "__main__":
    MODEL_PATH = "inclusionAI/Ming-Lite-Omni"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GENERATION_CONFIG = {
        "output_hidden_states": True,
        "return_dict_in_generate": True,
        "no_repeat_ngram_size": 10,
    }

    spk_info = {
            'luna': torch.load('data/spks/luna_v2.pt'),
            'eng': torch.load('data/spks/eng_v2.pt'),
        }

    processor, model, audio_detokenizer, audio_frontend = initialize_models(
        model_path=MODEL_PATH, spk_info=spk_info, device=DEVICE
    )
    model.talker.use_vllm = False

    # ASR
    print("Testing ASR...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {
                    "type": "text",
                    "text": "Please recognize the language of this speech and transcribe it. Format: oral.",
                },
                {"type": "audio", "audio": "data/wavs/BAC009S0915W0292.wav"},
            ],
        },
    ]

    generate(messages=messages, processor=processor, model=model)

    # Speech QA
    print("Testing Speech QA...")
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "audio", "audio": "data/wavs/speechQA_sample.wav"},
            ],
        },
    ]

    text_response, audio_wave = generate_e2e(
        messages=messages,
        processor=processor,
        model=model,
        audio_detokenizer=audio_detokenizer,
        output_audio_path="out.wav",
        spk_info=spk_info,
        generation_config=GENERATION_CONFIG,
        stream=False,
    )
    print(f"SPeech QA 文本响应：{text_response}")
    print("SPeech QA 音频响应已保存到 out.wav")

    # TTS Generation
    print("Testing TTS...")
    tts_wave = generate_tts(
        tts_text="这是一条测试语句。",
        prompt_text="感谢你的认可。",
        prompt_wav_path="data/spks/prompt.wav",
        audio_frontend=audio_frontend,
        model=model,
        audio_detokenizer=audio_detokenizer,
        output_audio_path="out_tts.wav",
        stream=False,
    )
    print("TTS 音频响应已保存到 out_tts.wav")
