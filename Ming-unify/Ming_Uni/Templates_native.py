# special tokens
DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>"
DEFAULT_IM_START_TOKEN = "<image>"
DEFAULT_IM_END_TOKEN = "</image>"
DEFAULT_VID_START_TOKEN = "<video>"
DEFAULT_VID_END_TOKEN = "</video>"
DEFAULT_GEN_IMAGE_PATCH_TOKEN = "<gen_imagePatch>"
DEFAULT_GEN_IM_START_TOKEN = "<gen_image>"
DEFAULT_GEN_IM_END_TOKEN = "</gen_image>"
PLACEHOLDER_IMAGE_TOKEN_IN_TEXT = "<imageHere>"
DEFAULT_END_OF_CHUNK_TOKEN = "<end_of_chunk>"

DEFAULT_END_OF_AUDIO_TOKEN = "<end_of_audio>"
DEFAULT_AUDIO_PATCH_TOKEN = "<audioPatch>"
DEFAULT_AU_START_TOKEN = "<audio>"
DEFAULT_AU_END_TOKEN = "</audio>"
DEFAULT_GEN_AUDIO_PATCH_TOKEN = "<gen_audioPatch>"
DEFAULT_GEN_AU_START_TOKEN = "<gen_audio>"
DEFAULT_GEN_AU_END_TOKEN = "</gen_audio>"
PLACEHOLDER_AUDIO_TOKEN_IN_TEXT = "<audioHere>"
DEFAULT_FRAME_PATCH_TOKEN = "<framePatch>"

interleave_tokens = [
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_GEN_IMAGE_PATCH_TOKEN,
    DEFAULT_GEN_IM_START_TOKEN,
    DEFAULT_GEN_IM_END_TOKEN,
    PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
    DEFAULT_END_OF_CHUNK_TOKEN,
    DEFAULT_END_OF_AUDIO_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_AU_START_TOKEN,
    DEFAULT_AU_END_TOKEN,
    DEFAULT_GEN_AUDIO_PATCH_TOKEN,
    DEFAULT_GEN_AU_START_TOKEN,
    DEFAULT_GEN_AU_END_TOKEN,
    PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
    DEFAULT_FRAME_PATCH_TOKEN
]


# prompts for qwen2
START_HEADER_QWEN2 = "<|im_start|>"
END_HEADER_QWEN2 = "<|im_end|>"
QWEN2_SYSTEM_PREFIX = "<|im_start|>system\nYou are a helpful assistant."
QWEN2_USER_PREFIX = "<|im_end|>\n<|im_start|>user\n"
QWEN2_ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"

# special tokens for llama3
START_HEADER = "<|start_header_id|>"  # Specifies the role for the following message, i.e. “system” 128006
END_HEADER = "<|end_header_id|>"  # 128007
EOT = "<|eot_id|>"  # Specifies the end of the input message [128009]
SYSTEM_PREFIX = START_HEADER + "system" + END_HEADER + "\n\n"  # system [128006, 9125, 128007, 271]
USER_PREFIX = START_HEADER + "user" + END_HEADER + "\n\n"  # user [128006, 882, 128007, 271]
ASSISTANT_PREFIX = START_HEADER + "assistant" + END_HEADER + "\n\n"  # assistant [128006, 78191, 128007, 271]

GLM_USER_PREFIX = "<role>HUMAN</role>"
GLM_ASSISTANT_PREFIX = "<role>ASSISTANT</role>"