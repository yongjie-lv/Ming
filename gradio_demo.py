import os
import torch
import uuid
import re

import gradio as gr
from gradio_client import utils as client_utils

from transformers import AutoProcessor, GenerationConfig
from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from modeling_bailing_talker import AudioDetokenizer


model_path = "inclusionAI/Ming-Lite-Omni"

# build model
model = BailingMMNativeForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    load_image_gen=True
).to("cuda")

# build processor
processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)


################################## demo utils ###################################

cache_dir = "demo_cache"
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)


def process_inputs(model, processor, messages, has_audio=False):
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    if has_audio:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
            audio_kwargs={'use_whisper_encoder': True}
        )
    else:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt"
        )
    
    inputs = inputs.to(model.device)
    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)
    
    return inputs


def generate_text(model, processor, messages, has_audio=False):
    inputs = process_inputs(model, processor, messages, has_audio=has_audio)

    # generation_config = GenerationConfig.from_dict({
    #     'output_hidden_states': True,
    #     'return_dict_in_generate': True,
    #     'no_repeat_ngram_size': 10}
    # )
    generation_config = GenerationConfig.from_dict({
        "no_repeat_ngram_size": 10
    })
    outputs = None
    if not has_audio:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            generation_config=generation_config
        )
    else:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            generation_config=generation_config,
            use_whisper_encoder=True
        )        

    # generated_ids = outputs.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text, outputs


def generate_image(model, processor, messages, has_audio=False):
    # Image generation mode currently limits the range of input pixels.
    gen_input_pixels = 451584
    processor.max_pixels = gen_input_pixels
    processor.min_pixels = gen_input_pixels

    inputs = process_inputs(model, processor, messages, has_audio=has_audio)

    image_gen_param = {
        "image_gen_cfg": 6.0,
        "image_gen_steps": 20,
        "image_gen_width": 512,
        "image_gen_height": 512
    }
    image = model.generate(
        **inputs,
        image_gen=True,
        **image_gen_param,
    )

    image_path = os.path.join(cache_dir, f"{uuid.uuid4()}.jpg")
    image.save(image_path)

    return image_path


def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def text_to_speach(model, output_text, outputs):
    model_name_or_path = model.config._name_or_path
    audio_detokenizer = AudioDetokenizer(
        f'{model_name_or_path}/talker/audio_detokenizer.yaml',
        flow_model_path=f'{model_name_or_path}/talker/flow.pt',
        hifigan_model_path=f'{model_name_or_path}/talker/hift.pt'
    )

    if contains_chinese(output_text):
        spk_input = torch.load('data/spks/luna.pt')
    else:
        spk_input = torch.load('data/spks/luna_eng.pt')

    if outputs is not None:
        thinker_reply_part = outputs.hidden_states[0][0] + outputs.hidden_states[0][-1]
    else:
        thinker_reply_part = None
    # Setting thinker_reply_part to None allows the talker to operate as a standalone TTS model, independent of the language model.
    audio_tokens = model.talker.omni_audio_generation(
        output_text, 
        thinker_reply_part=thinker_reply_part, **spk_input)
    audio_path = os.path.join(cache_dir, f"{uuid.uuid4()}.wav")
    waveform = audio_detokenizer.token2wav(audio_tokens, save_path=audio_path, **spk_input)

    return audio_path


def find_audio(messages):
    for msg in messages:
        for item in msg["content"]:
            if item["type"] == "audio":
                return True

    return False


def infer_user_intent(model, processor, user_input):  
    # question = "Based on the user instruction between <start_of_instruct> and <end_of_instruct> or provided in the audio, infer the actions to perform. If the task only needs to generate a text response, return a list of [\"text generation\"]. If the tasks only needs to generate an image, return a list of [\"image generation\"]. If the task needs to generate both a text response and an image, return a list of [\"text generation\", \"image generateion\"]. If no action needs to be performed, retuan []."
    question = "Based on the user instruction given between <instruct> and </instruct> or provided in the audio, infer the actions to perform. If the task only needs to generate a text response, return a list of [\"text generation\"]. If the tasks only needs to generate an image, return a list of [\"image generation\"]. If the task needs to generate both a text response and an image, return a list of [\"text generation\", \"image generateion\"]. If no action needs to be performed, retuan []."
    question = "According to the text above or the provided audio, infer the actions to perform. If the task only needs to generate a text response, return a list of [\"text generation\"]. If the tasks only needs to generate an image, return a list of [\"image generation\"]. If the task needs to generate both a text response and an image, return a list of [\"text generation\", \"image generateion\"]. If no action needs to be performed, retuan []."

    instruction = ""
    for item in user_input["content"]:
        if item["type"] == "text":
            instruction = instruction + item["text"] + "\n"
    # if len(instruction) > 0:
    #     instruction = "<start_of_instruct>\n" + instruction + "<end_of_instruct>\n"
    # if len(instruction) > 0:
    #     instruction = "<instruct>\n" + instruction + "</instruct>\n"

    audio_messages = [item for item in user_input["content"] if item["type"] == "audio"]
    has_audio = len(audio_messages) > 0

    output_text = None 
    if len(audio_messages) > 0 or len(instruction) > 0:
        content = []
        content.extend(audio_messages)
        question = instruction + question
        content.append({"type": "text", "text": question})       
        messages = [
            {
                "role": "HUMAN",
                "content": content
            }
        ]

        print("messages:")
        print(messages)

        output_text, _ = generate_text(model, processor, messages, has_audio=has_audio)

    try:
        outputs = output_text.replace("[", "").replace("]", "").replace("'", "").replace("\"", "").split(",")
        tasks = []
        for output in outputs:
            task = output.strip().lower()
            if "text generation" in task:
                tasks.append("text generation")
            elif "image generation" in task:
                tasks.append("image generation")
    except:
        tasks = ["text generation"]

    return tasks


def generate(model, processor, messages, state, use_audio_response=False):
    has_audio = find_audio(messages)

    # user_input = messages[-1]
    # tasks = infer_user_intent(model, processor, user_input)

    tasks = ["text_generation"]
    if state["gen_text"]:
        tasks = ["text generation"]
    elif state["gen_image"]:
        tasks = ["image generation"]

    text = audio_path = None
    if "text generation" in tasks:
        text, outputs = generate_text(model, processor, messages, has_audio=has_audio)

        if use_audio_response:
            audio_path = text_to_speach(model, text, outputs)

    image_path = None
    if "image generation" in tasks:
        image_path = generate_image(model, processor, messages, has_audio=has_audio)

    return text, audio_path, image_path
###########################################################################






def format_history(history: list):
    messages = []
    prev_role = "NONE"
    for message_pair in history:
        user_message, response = message_pair
        role = "HUMAN" if user_message is not None else "ASSISTANT"
        message = user_message if user_message is not None else response

        if isinstance(message, str):
            if role == prev_role:
                messages[-1]["content"].append({"type": "text", "text": message})
            else:
                messages.append({
                    "role": role, 
                    "content": [{"type": "text", "text": message}]
                })
        elif role == "HUMAN" and (isinstance(message, list) or isinstance(message, tuple)):
            file_path = message[0]
            mime_type = client_utils.get_mimetype(file_path)

            if mime_type.startswith("image"):
                if role == prev_role:
                    messages[-1]["content"].append({"type": "image", "image": file_path})
                else:
                    messages.append({
                        "role": role,
                        "content": [{"type": "image", "image": file_path}]
                    })
            elif mime_type.startswith("video"):
                if role == prev_role:
                    messages[-1]["content"].append({"type": "video", "video": file_path, "max_frames": 64, "sample": "uniform"})
                else:
                    messages.append({
                        "role": role,
                        "content": [{"type": "video", "video": file_path, "max_frames": 64, "sample": "uniform"}]
                    })
            elif mime_type.startswith("audio"):
                if role == prev_role:
                    messages[-1]["content"].append({"type": "audio", "audio": file_path})
                else:
                    messages.append({
                        "role": role,
                        "content": [{"type": "audio", "audio": file_path}]
                    })

        prev_role = role

    return messages


def chat_predict(text, audio, image, video, history, use_audio_response, state):
    # Process image input
    if image:
        # history.append({"role": "HUMAN", "content": (image, )})
        history.append(((image,), None))

    # Process video input
    if video:
        # history.append({"role": "HUMAN", "content": (video, )})
        history.append(((video,), None))

    # Process audio input
    if audio:
        # history.append({"role": "HUMAN", "content": (audio, )})
        history.append(((audio,), None))

    # Process text input
    if text:
        # history.append({"role": "HUMAN", "content": text})
        history.append((text, None))

    print(f"history: {history}")

    messages = format_history(history)

    yield None, None, None, None, history, gr.update(visible=False), gr.update(visible=True)

    print(messages)
    text, audio_path, image_path = generate(model, processor, messages, state, use_audio_response=use_audio_response)

    # print("Generation: done.")

    if text:
        history.append((None, text))

    if audio_path:
        history.append((None, (audio_path, )))

    if image_path:
        history.append((None, (image_path, )))

    yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history, gr.update(visible=True), gr.update(visible=False)


def fill_in_text_qa_example(text):
    return text, gr.update(value=None), gr.update(value=None), gr.update(value=None), []


def fill_in_image_qa_example(image, text):
    return text, gr.update(value=None), image, gr.update(value=None), []


def fill_in_video_qa_example(video, text):
    return text, gr.update(value=None), gr.update(value=None), video, []


def fill_in_asr_example(audio, text):
    return text, audio, gr.update(value=None), gr.update(value=None), []


def fill_in_speech_qa_example(audio):
    return gr.update(value=None), audio, gr.update(value=None), gr.update(value=None), [], True


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Ming-Lite-Omni Demo
        
        ## Instructions for use
        
        1. Upload an image, video or audio clip. 
        
        2. The instruction is input via the text or audio clip.

        3. Click on the Submit button and wait for the model's response.
        """
    )

    check_box = gr.Checkbox(label="Use audio response")

    chatbot = gr.Chatbot(type="tuples", height=650)

    # Media upload section in one row
    with gr.Row(equal_height=True):
        audio_input = gr.Audio(sources=["upload"],
                                type="filepath",
                                label="Upload Audio",
                                elem_classes="media-upload",
                                scale=1)
        image_input = gr.Image(sources=["upload"],
                                type="filepath",
                                label="Upload Image",
                                elem_classes="media-upload",
                                scale=1)
        video_input = gr.Video(sources=["upload"],
                                label="Upload Video",
                                elem_classes="media-upload",
                                scale=1)

    # Text input section
    text_input = gr.Textbox(show_label=False,
                            placeholder="Enter text here...")

    # Control buttons
    with gr.Row():
        gen_text_btn = gr.Button("Generate Text",
                                variant="primary",
                                size="lg")
        stop_text_gen_btn = gr.Button("Stop",
                             visible=False,
                             size="lg")
        gen_image_btn = gr.Button("Generate Image",
                                variant="primary",
                                size="lg")
        stop_image_gen_btn = gr.Button("Stop",
                             visible=False,
                             size="lg")

    clear_btn = gr.Button("Clear History",
                            size="lg")

    def clear_chat_history():
        return [], gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)

    text_gen_state = gr.State({"gen_text": True, "gen_image": False})

    gen_text_event = gen_text_btn.click(
        fn=chat_predict,
        inputs=[
            text_input, audio_input, image_input, video_input, chatbot, check_box, text_gen_state
        ],
        outputs=[
            text_input, audio_input, image_input, video_input, chatbot,  gen_text_btn, stop_text_gen_btn
        ])

    stop_text_gen_btn.click(fn=lambda:
                    (gr.update(visible=True), gr.update(visible=False)),
                    inputs=None,
                    outputs=[gen_text_btn, stop_text_gen_btn],
                    cancels=[gen_text_event],
                    queue=False)

    image_gen_state = gr.State({"gen_text": False, "gen_image": True})

    gen_image_event = gen_image_btn.click(
        fn=chat_predict,
        inputs=[
            text_input, audio_input, image_input, video_input, chatbot, check_box, image_gen_state
        ],
        outputs=[
            text_input, audio_input, image_input, video_input, chatbot, gen_image_btn, stop_image_gen_btn
        ])

    stop_image_gen_btn.click(fn=lambda:
                    (gr.update(visible=True), gr.update(visible=False)),
                    inputs=None,
                    outputs=[gen_image_btn, stop_image_gen_btn],
                    cancels=[gen_image_event],
                    queue=False)

    clear_btn.click(fn=clear_chat_history,
                    inputs=None,
                    outputs=[
                        chatbot, text_input, audio_input, image_input,
                        video_input
                    ])

    # Add some custom CSS to improve the layout
    gr.HTML("""
        <style>
            .media-upload {
                margin: 10px;
                min-height: 160px;
            }
            .media-upload > .wrap {
                border: 2px dashed #ccc;
                border-radius: 8px;
                padding: 10px;
                height: 100%;
            }
            .media-upload:hover > .wrap {
                border-color: #666;
            }
            /* Make upload areas equal width */""
            .media-upload {
                flex: 1;
                min-width: 0;
            }
        </style>
    """)

    gr.Markdown(
        """
        # Examples
        """
    )

    gr.Examples(
        fn=fill_in_text_qa_example,
        run_on_click=True,
        examples=[
            [
                "请详细介绍鹦鹉的生活习性"
            ],           
        ],
        label="Text QA",
        inputs=[text_input],
        outputs=[text_input, audio_input, image_input, video_input, chatbot]
    )

    gr.Examples(
        fn=fill_in_image_qa_example,
        run_on_click=True,
        examples=[
            [
                "figures/cases/flower.jpg",
                "What kind of flower is this?"
            ],           
        ],
        label="Image QA",
        inputs=[image_input, text_input],
        outputs=[text_input, audio_input, image_input, video_input, chatbot]
    )

    gr.Examples(
        fn=fill_in_video_qa_example,
        run_on_click=True,
        examples=[
            [
                "figures/cases/yoga.mp4",
                "What is the woman doing?"
                
            ],           
        ],
        label="Video QA",
        inputs=[video_input, text_input],
        outputs=[text_input, audio_input, image_input, video_input, chatbot]
    )

    gr.Examples(
        fn=fill_in_image_qa_example,
        run_on_click=True,
        examples=[
            [    
                "figures/cases/reasoning.png",
                "SYSTEM: You are a helpful assistant. When the user asks a question, your response must include two parts: first, the reasoning process enclosed in <thinking>...</thinking> tags, then the final answer enclosed in <answer>...</answer> tags. The critical answer or key result should be placed within \\boxed{}.\nPlease answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: Find $m\\angle H$\nChoices:\n(A) 97\n(B) 102\n(C) 107\n(D) 122.\n"
            ],
        ],
        label="Chain-of-Thought Reasoning",
        inputs=[image_input, text_input],
        outputs=[text_input, audio_input, image_input, video_input, chatbot]
    )

    gr.Examples(
        fn=fill_in_asr_example,
        run_on_click=True,
        examples=[
            [
                "data/wavs/BAC009S0915W0283.wav",
                "Please recognize the language of this speech and transcribe it. Format: oral."
                
            ],           
        ],
        label="Automatic speech recognition (ASR)",
        inputs=[audio_input, text_input],
        outputs=[text_input, audio_input, image_input, video_input, chatbot]
    )

    gr.Examples(
        fn=fill_in_speech_qa_example,
        run_on_click=True,
        examples=[
            [
                "data/wavs/speechQA_sample.wav"
            ]          
        ],
        label="Speech to speech",
        inputs=[audio_input],
        outputs=[text_input, audio_input, image_input, video_input, chatbot, check_box]
    )

    gr.Examples(
        fn=fill_in_text_qa_example,
        run_on_click=True,
        examples=[
            [
                "Draw a girl with short hair"
            ]          
        ],
        label="Image generation",
        inputs=[text_input],
        outputs=[text_input, audio_input, image_input, video_input, chatbot]
    )

    gr.Examples(
        fn=fill_in_image_qa_example,
        run_on_click=True,
        examples=[
            [
                "samples/cake.jpg",
                "Add a candle on top of the cake"
            ]          
        ],
        label="Image editing",
        inputs=[image_input, text_input],
        outputs=[text_input, audio_input, image_input, video_input, chatbot]
    )

demo.queue(max_size=100).launch(debug=True)

# from aistudio_notebook.ui import gradio
# gradio.launch(demo.queue(max_size=100), root_path="/proxy/libroApp_54990481:8080/proxy/app/7860", debug=True)
