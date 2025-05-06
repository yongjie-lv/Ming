import os
import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import DPMSolverMultistepScheduler, AutoencoderDC, FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from .qwen2_5_vit import Qwen2_5_VisionTransformer
from .modeling_qwen2_native import Qwen2ForCausalLM
from .sana_transformer import SanaTransformer2DModel
from .sana_loss import SANALoss
from copy import deepcopy
from IPython import embed

import logging
logger = logging.getLogger(__name__)

from .Templates_native import (
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
    DEFAULT_FRAME_PATCH_TOKEN,
    interleave_tokens,
)
additional_special_tokens_qwen2 = [
    "[item]",
    "<html>",
    "</html>",
    "<body>",
    "</body>",
    "<table>",
    "</table>",
    "<tr>",
    "</tr>",
    "<td>",
    "</td>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>"
]

def expand_gen_embeds_as_learnable_scales(
    clip_feat,
    image_grid_thw,
    scales,
    isgen_indicators,
    learnable_queries_1d,
):
    resized_clip_feat = []
    new_image_grid_thw = []
    
    assert image_grid_thw.ndim == 2
    bsz = len(image_grid_thw)
    assert clip_feat.ndim == 2
    feat_dim = clip_feat.shape[1]
    n_clip_token_cum = 0
    assert len(isgen_indicators) == bsz
    #assert image_grid_thw.ndim == 3
    for bsid in range(bsz):
        thw = image_grid_thw[bsid].tolist()
        assert thw[0] == 1
        assert thw[1] % 2 == 0
        assert thw[2] % 2 == 0
        clip_h = thw[1] // 2
        clip_w = thw[2] // 2
        n_clip_token = clip_h * clip_w
        assert n_clip_token_cum + n_clip_token <= clip_feat.shape[0]
        if isgen_indicators[bsid]:
            for scale in scales:
                clip_feat_one = torch.zeros(scale * scale, feat_dim).to(clip_feat.dtype).to(clip_feat.device)
                resized_clip_feat.append(clip_feat_one)
                if learnable_queries_1d:
                    new_image_grid_thw.append([1, 2, scale * scale * 2])
                else:
                    new_image_grid_thw.append([1, scale * 2, scale * 2])
        else:
            clip_feat_one = clip_feat[n_clip_token_cum : n_clip_token_cum + n_clip_token, :]
            resized_clip_feat.append(clip_feat_one)
            new_image_grid_thw.append(thw)

        n_clip_token_cum += n_clip_token
    
    assert n_clip_token_cum == clip_feat.shape[0]

    encoder_hidden_states = torch.cat(resized_clip_feat, dim=0)
    return encoder_hidden_states, torch.tensor(new_image_grid_thw, dtype=image_grid_thw.dtype).to(image_grid_thw.device)

def append_understand_embeds_with_learnable_scales(
    clip_feat,
    image_grid_thw,
    scales,
    dtype,
    device,
    feat_dim,
    learnable_queries_1d,
):
    if clip_feat is not None:
        assert feat_dim == clip_feat.shape[-1]
        assert dtype == clip_feat.dtype
        assert device == clip_feat.device
        assert clip_feat.ndim == 2
    else:
        assert image_grid_thw is None
    
    fake_learnable_embed = torch.zeros(256, feat_dim).to(dtype).to(device)
    clip_feat = torch.cat([clip_feat, fake_learnable_embed], dim=0) if clip_feat is not None else fake_learnable_embed
    fake_image_grid_thw = torch.tensor([[1, 32, 32]], dtype=torch.long).to(device)
    image_grid_thw = torch.cat([image_grid_thw, fake_image_grid_thw], dim=0) if image_grid_thw is not None else fake_image_grid_thw

    return expand_gen_embeds_as_learnable_scales(
        clip_feat,
        image_grid_thw,
        scales,
        isgen_indicators=[False for _ in range(image_grid_thw.shape[0]-1)] + [True],
        learnable_queries_1d=learnable_queries_1d,
    )

def expand_gen_input_ids_as_learnable_scales(
    text_ids,
    labels,
    attention_mask,
    scales,
    start_token_id,
    end_token_id,
    patch_token_id,
    num_learnable_queries,
):
    assert text_ids.ndim == 2
    assert text_ids.shape == labels.shape
    assert text_ids.shape == attention_mask.shape

    default_scaled_tokens = []
    for scale in scales:
        default_scaled_tokens.append(start_token_id)
        default_scaled_tokens.extend([patch_token_id for _ in range(scale * scale)])
        default_scaled_tokens.append(end_token_id)
    
    text_ids_list = text_ids.cpu().tolist()
    labels_list = labels.cpu().tolist()
    attention_mask_list = attention_mask.cpu().tolist()

    new_text_ids_list = []
    new_labels_list = []
    new_attention_mask_list = []
    for text_ids_one_batch, labels_one_batch, attention_mask_one_batch in zip(text_ids_list, labels_list, attention_mask_list):
        assert len(text_ids_one_batch) == len(labels_one_batch)
        assert len(text_ids_one_batch) == len(attention_mask_one_batch)
        start_idx = [i for i, j  in enumerate(labels_one_batch) if j == start_token_id]
        end_idx = [i for i, j in enumerate(labels_one_batch) if j == end_token_id]
        assert len(start_idx) == 1, start_idx
        assert len(end_idx) == 1, end_idx
        start_idx = start_idx[0]
        end_idx = end_idx[0]
        assert end_idx - start_idx == num_learnable_queries + 1, (start_idx, end_idx)
        assert text_ids_one_batch[start_idx] == start_token_id and text_ids_one_batch[end_idx] == end_token_id
        text_ids_one_batch[start_idx: end_idx+1] = deepcopy(default_scaled_tokens)
        labels_one_batch[start_idx: end_idx+1] = deepcopy(default_scaled_tokens)
        attention_mask_one_batch[start_idx: end_idx+1] = [1 for _ in range(len(default_scaled_tokens))]

        new_text_ids_list.append(text_ids_one_batch)     
        new_labels_list.append(labels_one_batch) 
        new_attention_mask_list.append(attention_mask_one_batch)       

    return (
        torch.tensor(new_text_ids_list, dtype=text_ids.dtype).to(text_ids.device),
        torch.tensor(new_labels_list, dtype=labels.dtype).to(labels.device), 
        torch.tensor(new_attention_mask_list, dtype=attention_mask.dtype).to(attention_mask.device)
    )


def append_input_ids_with_learnable_scales(    
    text_ids,
    scales,
    start_token_id,
    end_token_id,
    patch_token_id,
):
    assert text_ids.shape[0] == 1
    assert text_ids[0][-1].tolist() == start_token_id

    labels = torch.cat([
        torch.ones_like(text_ids[:,:-1]) * 0 - 100, 
        torch.tensor([[start_token_id, patch_token_id, end_token_id]]).to(text_ids.dtype).to(text_ids.device),
    ], dim=1)
    
    text_ids = torch.cat([
        text_ids, 
        torch.tensor([[patch_token_id, end_token_id]]).to(text_ids.dtype).to(text_ids.device),
    ], dim=1)

    assert labels.shape == text_ids.shape

    attention_mask = torch.ones_like(text_ids)
    text_ids, labels, attention_mask = expand_gen_input_ids_as_learnable_scales(
        text_ids,
        labels,
        attention_mask,
        scales,
        start_token_id,
        end_token_id,
        patch_token_id,
        num_learnable_queries=1,
    )
    return text_ids, labels

class Ming_Uni_Inference(nn.Module):
    def __init__(self, inference_model_path):
        super(Ming_Uni_Inference, self).__init__()
        self.inference_model_path = inference_model_path
        print('loading from pretrained:',inference_model_path)
        self.load_from_huggingface()
        #embed()

    def init_tokens(self):
        num_query_token=2560
        num_query_token_video=64
        num_query_token_audio=32
        num_decoder_image_token=1024
        num_decoder_audio_token=512
        self.glm_tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens_qwen2}
        )
        num_new_tokens = self.glm_tokenizer.add_tokens(
            interleave_tokens,
            special_tokens=True,
        )
        logger.warning("init_mm_specail_tokens: generation_num_tokens = {}".format(num_new_tokens))
        self.glm_config.first_signal_token = self.glm_tokenizer.convert_tokens_to_ids("[IMG0]")
        self.glm_config.image_start_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        self.glm_config.image_end_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        self.glm_config.image_patch_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)
        self.glm_config.video_start_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_VID_START_TOKEN)
        self.glm_config.video_end_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_VID_END_TOKEN)
        self.glm_config.gen_image_start_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_GEN_IM_START_TOKEN)
        self.glm_config.gen_image_end_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_GEN_IM_END_TOKEN)
        self.glm_config.gen_image_patch_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_GEN_IMAGE_PATCH_TOKEN)
        self.glm_config.placeholder_image_token_in_text = self.glm_tokenizer.convert_tokens_to_ids(
            PLACEHOLDER_IMAGE_TOKEN_IN_TEXT
        )  # noqa
        self.glm_config.end_of_chunk_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_END_OF_CHUNK_TOKEN)

        self.glm_config.end_of_audio_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_END_OF_AUDIO_TOKEN)
        self.glm_config.audio_start_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_AU_START_TOKEN)
        self.glm_config.audio_end_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_AU_END_TOKEN)
        self.glm_config.audio_patch_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_PATCH_TOKEN)
        self.glm_config.gen_audio_start_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_GEN_AU_START_TOKEN)
        self.glm_config.gen_audio_end_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_GEN_AU_END_TOKEN)
        self.glm_config.gen_audio_patch_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_GEN_AUDIO_PATCH_TOKEN)
        self.glm_config.placeholder_audio_token_in_text = self.glm_tokenizer.convert_tokens_to_ids(
            PLACEHOLDER_AUDIO_TOKEN_IN_TEXT
        )  # noqa
        self.glm_config.frame_patch_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_FRAME_PATCH_TOKEN)
        self.glm_config.video_patch_token = self.glm_tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)

        self.glm_config.num_image_token = num_query_token
        self.glm_config.num_video_token = num_query_token_video
        self.glm_config.num_audio_token = num_query_token_audio
        self.glm_config.num_decoder_image_token = num_decoder_image_token
        self.glm_config.num_decoder_audio_token = num_decoder_audio_token

    def load_from_huggingface(self):
        # Load Qwen2_5_vit
        self.eva_encoder = Qwen2_5_VisionTransformer.from_pretrained(
            os.path.join(self.inference_model_path, 'qwen2_5_vit'),
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            force_download=True,
        ) 
        
        # Load Qwen2_5_llm (GLM model)


        self.glm_tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.inference_model_path, 'qwen2_5_llm'))
        self.glm_config = Qwen2ForCausalLM.from_pretrained(os.path.join(self.inference_model_path, 'qwen2_5_llm')).config
        
        self.init_tokens()
        self.glm_config.audio_vocab_size = 4099
        self.glm_config.audio_id_shift = 151699
        self.glm_config.spatial_merge_size = 2
        self.glm_config.tokens_per_second = 2
        self.glm_config._attn_implementation = "flash_attention_2"
        self.glm_config.use_llm_3drope = True        
        self.glm_model = Qwen2ForCausalLM.from_pretrained(os.path.join(self.inference_model_path, 'qwen2_5_llm'), config=self.glm_config)

        # Load SANA
        # self.scheduler = DPMSolverMultistepScheduler.from_pretrained(self.inference_model_path, subfolder="scheduler")
        # self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.inference_model_path, subfolder="scheduler")
        # self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        # self.vae = AutoencoderDC.from_pretrained(self.inference_model_path, subfolder="vae")
        # self.train_model = SanaTransformer2DModel.from_pretrained(self.inference_model_path, subfolder="transformer")
        # self.train_model = SanaModel_withMLP(self.train_model, vision_dim=self.glm_model.config.hidden_size)  # Ensure vision_dim is properly defined/set
        # mlp_checkpoint_path = os.path.join(self.inference_model_path, 'mlp', 'model.safetensors')
        # assert os.path.exists(mlp_checkpoint_path), "MLP checkpoint path does not exist."
        # inference_load_denoising_pretrained_weights(self.train_model, mlp_checkpoint_path)
        self.diffloss = SANALoss(
            model_path=self.inference_model_path, 
            scheduler_path=self.inference_model_path, 
            vision_dim=self.glm_model.config.hidden_size, 
            mlp_checkpoint_path=os.path.join(self.inference_model_path, 'mlp', 'model.safetensors'),
            trainable_params="",
        )
        # Load MLP
        self.image_emb_dim = 8192
        mlp_modules_img = [nn.Linear(self.image_emb_dim, self.glm_model.config.hidden_size)]
        for _ in range(1, 2):
            mlp_modules_img.append(nn.GELU())
            mlp_modules_img.append(nn.Linear(self.glm_model.config.hidden_size, self.glm_model.config.hidden_size))
        self.linear_proj = nn.Sequential(*mlp_modules_img)
        temp_state_dict = load_file(os.path.join(self.inference_model_path, 'mlp', 'model.safetensors'))
        modified_state_dict = {
                            '0.weight': temp_state_dict['linear_proj.0.weight'],
                            '0.bias': temp_state_dict['linear_proj.0.bias'],
                            '2.weight': temp_state_dict['linear_proj.2.weight'],
                            '2.bias': temp_state_dict['linear_proj.2.bias']
                        }
        self.linear_proj.load_state_dict(modified_state_dict, strict=True)
        self.norm_query_embeds = True
        # Load connector
        self.connector = AutoModelForCausalLM.from_pretrained(os.path.join(self.inference_model_path, 'connector'))
        for layer in self.connector.model.layers:
            layer.self_attn.is_causal = False
        
        self.proj_in = nn.Linear(self.glm_model.config.hidden_size, self.connector.config.hidden_size)
        self.proj_out = nn.Linear(self.connector.config.hidden_size, self.glm_model.config.hidden_size)

        temp_state_dict = load_file(os.path.join(self.inference_model_path, 'mlp', 'model.safetensors'))
        modified_state_dict_in = {
            'weight': temp_state_dict['proj_in.weight'],
            'bias': temp_state_dict['proj_in.bias']
        }
        self.proj_in.load_state_dict(modified_state_dict_in, strict=True)

        modified_state_dict_out = {
            'weight': temp_state_dict['proj_out.weight'],
            'bias': temp_state_dict['proj_out.bias']
        }
        self.proj_out.load_state_dict(modified_state_dict_out, strict=True)

        self.num_learnable_queries = 256
        self.use_multi_scale = True
        self.scales = [4, 8, 16]
        self.learnable_queries_1d = True
        
        
        self.query_tokens_dict = nn.ParameterDict()
        total_tokens = 0
        for scale in self.scales:                    
            num_tokens = scale * scale
            self.query_tokens_dict[f"{scale}x{scale}"] = nn.Parameter(
                torch.nn.functional.normalize(torch.randn(num_tokens, self.glm_model.config.hidden_size), dim=-1)
            )
            self.query_tokens_dict[f"{scale}x{scale}"].data = temp_state_dict[f"query_tokens_dict.{scale}x{scale}"]
            total_tokens += num_tokens
        
        # 计算各尺度的累积索引
        self.scale_indices = []
        current_idx = 0
        for scale in self.scales:
            current_idx += scale * scale
            self.scale_indices.append(current_idx)

        logger.info("All models load done.")

    @torch.no_grad()
    def image_gen_generate(
        self,
        samples,
        steps=20,
        seed=42,
        cfg=7.0,
        height=512,
        width=512,
        num_max_output_tokens=100,
    ):
        """
        Args:
            samples (dict): A dictionary containing the output of processor
            steps (int): Number of inference steps for diffusion
            height (int): height for output image
            width (int): width for output image
        Returns:
            result_word (str): output words
            result_image (PIL.Image): output image
        """

        assert samples["input_ids"].ndim == 2
        assert samples["input_ids"].shape[0] == 1
        if samples["input_ids"][0][-1].tolist() != self.glm_config.image_start_token:
            print("Warning: No <image> found at the end of prompt, back to chat mode.")

        image_embed_list = []
        if ("image" in samples) and (samples["image"] is not None):
            device = samples["image"].device
            images = samples["image"]
            if not isinstance(images, list):
                images = [images]
        else:
            device = samples["input_ids"].device
            images = []

        image_embed_list = []
        image_grid_thw = None
        for idx, item in enumerate(images):
            if len(images) > 0 and images[idx].size(0) > 0:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    pixel_values = images[idx].type(self.eva_encoder.get_dtype())
                    image_grid_thw = samples["image_grid_thw"]
                    eva_image_feat = self.eva_encoder(pixel_values, grid_thw=image_grid_thw)

                image_embed_list.append(eva_image_feat)

        image_embeds = None
        inputs_opt_visual = None
        device = samples["input_ids"].device
        if len(image_embed_list) > 0:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                image_embeds = torch.cat(image_embed_list).to(device)
                image_embeds = image_embeds.float()
                
                inputs_opt_visual = self.linear_proj(image_embeds)
                
            if self.norm_query_embeds:
                inputs_opt_visual = torch.nn.functional.normalize(inputs_opt_visual, dim=-1)  
            else:
                inputs_opt_visual = inputs_opt_visual * self.query_embeds_scale

            # if self.half_glm:
            #     inputs_opt_visual = inputs_opt_visual.half()
        

        inputs = {}
        inputs["input_ids"] = samples["input_ids"].to(device)
        assert "position_ids" not in samples or samples["position_ids"] is None
        inputs["position_ids"] = None
        inputs["attention_mask"] = samples["generation_attention_mask"].to(device)
 
        query_embeds_image = inputs_opt_visual
        query_embeds_video = None
        image_grid_thw_video = None
        inputs["query_embeds_image"] = query_embeds_image
        inputs["query_embeds_video"] = query_embeds_video
        inputs["image_grid_thw"] = image_grid_thw
        inputs["image_grid_thw_video"] = image_grid_thw_video

        output_str = ""
        new_token_ids = None
        new_query_embeds_images = None
        assert inputs["input_ids"].shape[0] == 1
        assert inputs["position_ids"] is None

        num_remaining_image_gen_token = 0
        curr_image_grid_thw = inputs["image_grid_thw"]
        for _ in range(num_max_output_tokens):
            assert num_remaining_image_gen_token >= 0
            curr_input_ids = torch.cat([inputs["input_ids"], new_token_ids], dim=1) if new_token_ids is not None else inputs["input_ids"]
            assert num_remaining_image_gen_token >= 0
            true_input_ids = curr_input_ids if num_remaining_image_gen_token == 0 else curr_input_ids[:,:-1 * (num_remaining_image_gen_token + 1)]

            curr_query_embeds_image = inputs["query_embeds_image"]
            if new_query_embeds_images is not None:
                if curr_query_embeds_image is None:
                    curr_query_embeds_image = new_query_embeds_images
                else:
                    curr_query_embeds_image = torch.cat([
                        curr_query_embeds_image,
                        new_query_embeds_images
                    ], dim=0)   
            
            if true_input_ids[0][-1].tolist() == self.glm_config.image_start_token:
                assert num_remaining_image_gen_token == 0
                apppended_query_embeds_image, curr_image_grid_thw = append_understand_embeds_with_learnable_scales(
                    clip_feat=curr_query_embeds_image,
                    image_grid_thw=curr_image_grid_thw,
                    scales=self.scales,
                    dtype=torch.bfloat16,
                    device=device,
                    feat_dim=self.glm_model.config.hidden_size,
                    learnable_queries_1d=self.learnable_queries_1d,
                )
                curr_input_ids, labels = append_input_ids_with_learnable_scales(
                    text_ids=true_input_ids,
                    scales=self.scales,
                    start_token_id=self.glm_model.config.image_start_token,
                    end_token_id=self.glm_model.config.image_end_token,
                    patch_token_id=self.glm_model.config.image_patch_token,
                )

                learnable_queries_repeat = torch.cat(
                    [self.query_tokens_dict[f"{scale}x{scale}"] for scale in self.scales], 
                    dim=0,
                )
                
                # 现在基于更新后的text_ids和labels计算inner_gen_mask
                image_token_mask = (curr_input_ids == self.glm_model.config.image_patch_token).to(device)
                inner_gen_mask = torch.masked_select(labels, image_token_mask) == self.glm_model.config.image_patch_token
                inner_gen_mask = inner_gen_mask.unsqueeze(-1).expand_as(apppended_query_embeds_image).to(apppended_query_embeds_image.device)

                apppended_query_embeds_image = apppended_query_embeds_image.masked_scatter(
                    inner_gen_mask, 
                    learnable_queries_repeat
                )
                assert new_token_ids is None
                new_token_ids = curr_input_ids[:, true_input_ids.shape[1]:]
                assert new_query_embeds_images is None
                new_query_embeds_images = apppended_query_embeds_image[curr_query_embeds_image.shape[0]:, :] if curr_query_embeds_image is not None else apppended_query_embeds_image
            
                continue
            
            curr_position_ids = self.glm_model.get_rope_index(curr_input_ids, curr_image_grid_thw)[0]    
            true_position_ids = curr_position_ids[:,:,:true_input_ids.shape[1]]

            outputs = self.glm_model(
                input_ids=true_input_ids,
                query_embeds_image=curr_query_embeds_image, 
                query_embeds_video=inputs["query_embeds_video"],
                query_embeds_audio=None, 
                target_embeds=None,  
                position_ids=true_position_ids,
                attention_mask=None,
                labels=None,
                weights=None,
                image_grid_thw=curr_image_grid_thw,
                image_grid_thw_video=image_grid_thw_video,
            )

            if new_query_embeds_images is not None:
                assert labels.shape == true_input_ids.shape
                gen_image_mask = labels == self.glm_model.config.image_patch_token
                assert gen_image_mask.sum().cpu().item() == new_query_embeds_images.shape[0]
                hidden_states_gen = outputs.last_hidden_state[gen_image_mask].view(outputs.last_hidden_state.shape[0], -1, outputs.last_hidden_state.shape[-1])
                assert hidden_states_gen.shape[1] == new_query_embeds_images.shape[0]
                scale_start_idxes = [0] + self.scale_indices[:-1]
                scale_end_idxes = self.scale_indices
                assert scale_end_idxes[-1] == hidden_states_gen.shape[1]
                new_query_embeds_images = {}
                for scale, scale_start_idx, scale_end_idx in zip(self.scales, scale_start_idxes, scale_end_idxes):   
                    scale_name = f"{scale}x{scale}"
                    scale_hidden = hidden_states_gen[:, scale_start_idx : scale_end_idx, :]

                    
                    scale_embeds = self.proj_in(scale_hidden)
                    seq_shape = scale_embeds.shape
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        scale_embeds = self.connector(
                            inputs_embeds=scale_embeds, 
                            attention_mask=torch.ones(seq_shape[0],1,seq_shape[1],seq_shape[1]).to(scale_embeds.device), 
                            output_hidden_states=True
                        ).hidden_states[-1]
                    scale_embeds = self.proj_out(scale_embeds)
                    
                    
                    scale_embeds = torch.nn.functional.normalize(scale_embeds, dim=-1)
                    new_query_embeds_images[scale_name] = scale_embeds
                
                break
            
            assert num_remaining_image_gen_token == 0
            new_token_id = outputs.logits[:,-1:,:].argmax(dim=-1)
            if (new_token_id.tolist())[0][0] == self.eos_token_id:
                break
            
            new_token_ids = torch.cat([new_token_ids, new_token_id], dim=1) if new_token_ids is not None else new_token_id
            output_str = output_str + self.glm_tokenizer.decode(new_token_id.tolist()[0])
        
        #multiscale_result = None
        if self.diffloss is not None and new_query_embeds_images is not None:
            #print("curr_image_grid_thw: ", curr_image_grid_thw)
            imgs = []
            for scale in self.scales:
                imgs.append(self.diffloss.sample(new_query_embeds_images[f"{scale}x{scale}"], steps=steps, seed=seed, cfg=cfg, height=height, width=width))
            
            #multiscale_result = concat_horizontal(imgs)
            new_query_embeds_images = imgs[-1]
            
        # if self.use_multi_scale:
        #     return output_str, new_query_embeds_images, multiscale_result
    
        return output_str, new_query_embeds_images

# Usage example:
# from MingUniInference import Ming_Uni_Inference
# model = Ming_Uni_Inference('/videomm/share/models/xinyu/test1')
