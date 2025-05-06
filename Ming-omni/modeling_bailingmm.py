#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging
from configuration_bailingmm import BailingMMConfig
from modeling_utils import patch_continuous_features

# audio encoder
from funasr.models.sanm.encoder import SANMEncoder
from modeling_bailing_moe import BailingMoeForCausalLM
from modeling_utils import Transpose, encode_audio_segments

# vision encoder
from qwen2_5_vit import Qwen2_5_VisionTransformer

# talker
from modeling_bailing_talker import BailingTalkerForConditionalGeneration

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BailingMMConfig"


@dataclass
class BailingMMCausalLMOutputWithPast(ModelOutput):
    """
    Base class for BailingMM causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class BailingMMNativeForConditionalGeneration(PreTrainedModel):
    config_class = BailingMMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingAudioModel"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(
        self,
        config: BailingMMConfig,
    ):
        super().__init__(config)
        self.config: BailingMMConfig = config
        self.vision = None
        self.audio = None
        self.talker = None

        self.llm_dytpe = torch.bfloat16

        if self.config.vision_config:
            self.vision = Qwen2_5_VisionTransformer(self.config.vision_config)

        if self.config.audio_config:
            self.audio = SANMEncoder(**self.config.audio_config.audio_encoder_config_sanm)

        self.model = BailingMoeForCausalLM(self.config.llm_config)

        mlp_modules_img = [nn.Linear(self.vision.image_emb_dim, self.model.config.hidden_size)]
        for _ in range(1, self.config.mlp_depth):
            mlp_modules_img.append(nn.GELU())
            mlp_modules_img.append(nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size))
        self.linear_proj = nn.Sequential(*mlp_modules_img)

        if self.audio:
            audio_encoder_proj = torch.nn.Conv1d(
                self.config.audio_config.audio_encoder_output_size,
                self.model.config.hidden_size,
                kernel_size=self.config.audio_config.ds_kernel_size,
                stride=self.config.audio_config.ds_stride,
                padding=self.config.audio_config.ds_kernel_size // 2,
            )

            mlp_modules_audio = [audio_encoder_proj, Transpose(-1, -2)]
            for _ in range(1, self.config.mlp_depth):
                mlp_modules_audio.append(nn.GELU())
                mlp_modules_audio.append(nn.Linear(
                    self.model.config.hidden_size, self.model.config.hidden_size
                ))
            mlp_modules_audio.append(Transpose(-1, -2))
            self.linear_proj_audio = nn.Sequential(*mlp_modules_audio)

        if self.config.talker_config:
            self.config.talker_config._name_or_path = f'{self.config._name_or_path}/talker'
            self.talker = BailingTalkerForConditionalGeneration(self.config.talker_config)

        self.post_init()

    def extract_image_feature(self, pixel_values, grid_thw):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            image_embeds = self.vision(pixel_values, grid_thw=grid_thw)
            image_embeds = image_embeds.float()
            image_embeds = self.linear_proj(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds
    
    def extract_audio_feature(self, audio_feats, audio_feats_lengths):
        assert self.audio is not None
        assert self.linear_proj_audio is not None
        audio_embeds, _, audio_embeds_lengths = encode_audio_segments(
            encoder=self.audio,
            proj_layer=self.linear_proj_audio,
            wav_feats=audio_feats,
            wav_feats_lengths=audio_feats_lengths,
        )
        if self.config.audio_config.norm_query_embeds:
            audio_embeds = F.normalize(audio_embeds, dim=2)  # [-1, 256, 2048]
        return audio_embeds, audio_embeds_lengths

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        audio_feats: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_feats_lengths: Optional[torch.LongTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        **generate_kwargs,
    ):
        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None
        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
        if pixel_values_videos is not None:
            video_embeds = self.extract_image_feature(pixel_values_videos, grid_thw=video_grid_thw)
        if audio_feats is not None:
            audio_embeds, audio_embeds_lengths = self.extract_audio_feature(audio_feats, audio_feats_lengths)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if (image_embeds is None and video_embeds is None and audio_embeds is None) or input_ids.size(1) == 1:
                words_embeddings = self.model.get_input_embeddings()(input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1))
                # input_shape = input_ids.size()
                batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
                image_start_indices = [999999] * batch_size
                image_end_indices = [999999] * batch_size
                audio_start_indices = [999999] * batch_size
                audio_end_indices = [999999] * batch_size

            else:
                words_embeddings, image_start_indices, image_end_indices, audio_start_indices, audio_end_indices = self.prompt_wrap_navit(
                        input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1), image_embeds, video_embeds, audio_embeds,
                        audio_embeds_lengths, audio_placeholder_loc_lens, None,  # noqa
                )

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=words_embeddings,
                use_cache=use_cache,
                image_start_indices=image_start_indices,
                image_end_indices=image_end_indices,
                audio_start_indices=audio_start_indices,
                audio_end_indices=audio_end_indices,
                **generate_kwargs,
            )
        return outputs

    def prompt_wrap_vision(self, input_ids, inputs_embeds, vision_embeds, image_token_id=None):
        if vision_embeds is None or input_ids is None:
            return inputs_embeds

        if len(vision_embeds.shape) == 3:
            vision_embeds = vision_embeds.reshape(-1, vision_embeds.shape[-1])

        self.config.llm_config.image_patch_token = image_token_id if image_token_id is not None else self.config.llm_config.image_patch_token
        n_image_tokens = (input_ids == self.config.llm_config.image_patch_token).sum().item()
        n_image_features = vision_embeds.shape[0]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        seq_len = input_ids.shape[1]
        is_image_token = (input_ids == self.config.llm_config.image_patch_token).int()
        first_indices = torch.argmax(is_image_token, dim=-1) - 1
        last_indices_flipped = torch.argmax( is_image_token.flip(dims = [1]), dim=-1)
        last_indices = (seq_len - 0) - last_indices_flipped

        image_mask = (
            (input_ids == self.config.llm_config.image_patch_token)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds,  first_indices.reshape(-1).tolist(), last_indices.reshape(-1).tolist()

    def prompt_wrap_audio(self, input_ids, inputs_embeds, audio_embeds, audio_embeds_lengths, placeholder_audio_loc_lens):
        assert placeholder_audio_loc_lens.shape[1] == 1, f"Currently MoE models do not support multiple audios in a single sample, but placeholder_audio_loc_lens = {placeholder_audio_loc_lens}"
        inputs_embeds = patch_continuous_features(
           input_embeddings=inputs_embeds, placeholder_loc_lens=placeholder_audio_loc_lens,
           encoded_feats=audio_embeds, encoded_feat_lens=audio_embeds_lengths,
        )
        first_indices = placeholder_audio_loc_lens[:, 0, 0] - 1
        last_indices = placeholder_audio_loc_lens[:, 0, 0] + placeholder_audio_loc_lens[:, 0, 1]
        return inputs_embeds, first_indices.reshape(-1).tolist(), last_indices.reshape(-1).tolist()
     
    def prompt_wrap_navit(self, input_ids, query_embeds_image=None, query_embeds_video=None, query_embeds_audio=None,
        query_embeds_audio_lengths=None, placeholder_audio_loc_lens=None, target_embeds=None):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if query_embeds_image is None and query_embeds_video is None and query_embeds_audio is None and target_embeds is None:
            return inputs_embeds

        audio_start_indices_list = None
        audio_end_indices_list = None
        image_start_indices_list = None
        image_end_indices_list = None
      
        batch_size = input_ids.shape[0]

        if query_embeds_image is not None:
            inputs_embeds, image_start_indices_list, image_end_indices_list = self.prompt_wrap_vision(input_ids, inputs_embeds, query_embeds_image)
        if query_embeds_video is not None:
            inputs_embeds, image_start_indices_list, image_end_indices_list = self.prompt_wrap_vision(input_ids, inputs_embeds, query_embeds_video)
        if query_embeds_audio is not None:
            inputs_embeds, audio_start_indices_list, audio_end_indices_list = self.prompt_wrap_audio(
                input_ids, inputs_embeds, query_embeds_audio, query_embeds_audio_lengths, placeholder_audio_loc_lens,
            )

        if audio_start_indices_list is None: audio_start_indices_list = [99999] * batch_size
        if audio_end_indices_list is None: audio_end_indices_list = [99999] * batch_size
        if image_start_indices_list is None: image_start_indices_list = [99999] * batch_size
        if image_end_indices_list is None: image_end_indices_list = [99999] * batch_size

        return inputs_embeds, image_start_indices_list, image_end_indices_list, audio_start_indices_list, audio_end_indices_list

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        audio_feats: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_feats_lengths: Optional[torch.LongTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple, BailingMMCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if (pixel_values is not None or pixel_values_videos is not None or audio_feats is not None) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos/pixel_values_audios and inputs_embeds at the same time, and must specify either one"
            )
        
        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None
        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
        if pixel_values_videos is not None:
            video_embeds = self.extract_image_feature(pixel_values_videos, grid_thw=video_grid_thw)
        if audio_feats is not None:
            audio_embeds, audio_embeds_lengths = self.extract_audio_feature(audio_feats, audio_feats_lengths)

        if (image_embeds is None and video_embeds is None and audio_embeds is None) or input_ids.size(1) == 1:
            words_embeddings = self.model.get_input_embeddings()(input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1))
            batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
            image_indices = [999999] * batch_size
            image_end_indices = [999999] * batch_size
            audio_indices = [999999] * batch_size
            audio_end_indices = [999999] * batch_size

        else:
            words_embeddings, image_indices, image_end_indices, audio_indices, audio_end_indices = self.prompt_wrap_navit(
                    input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1), image_embeds, video_embeds, audio_embeds,
                    audio_embeds_lengths, audio_placeholder_loc_lens, None,  # noqa
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=words_embeddings,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_indices=image_indices,
            image_end_indices=image_end_indices,
            audio_indices=audio_indices,
            audio_end_indices=audio_end_indices,
        )

        return BailingMMCausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
