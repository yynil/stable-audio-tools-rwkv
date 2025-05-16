from __future__ import annotations

import torch
import torch.nn as nn
from rwkvfla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from .rwkv7 import ContinuousRWKV
from typing import Optional, Tuple, Union, Literal
import math
from einops import rearrange
import torch.nn.functional as F
import logging
import os
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG if os.environ.get("DEBUG", "false").lower() == "true" else logging.WARNING)

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(
            [out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class DiffusionRWKV7(nn.Module):
    def __init__(
        self,
        config: RWKV7Config,
        *,
        io_channels: int = 32,
        patch_size: int = 1,
        cond_token_dim: int = 0,
        project_cond_tokens: bool = True,
        global_cond_dim: int = 0,
        project_global_cond: bool = True,
        input_concat_dim: int = 0,
        prepend_cond_dim: int = 0,
        global_cond_type: Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: Literal["global", "input_concat"] = "global",
        timestep_embed_dim: Optional[int] = None,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        **kwargs
    ):
        super().__init__()
        
        self.cond_token_dim = cond_token_dim
        self.timestep_cond_type = timestep_cond_type
        self.global_cond_type = global_cond_type
        self.diffusion_objective = diffusion_objective
        
        # 时间步特征
        timestep_features_dim = 256
        self.timestep_features = FourierFeatures(1, timestep_features_dim)
        
        if timestep_cond_type == "global":
            timestep_embed_dim = config.hidden_size
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim
            
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        
        # 条件标记处理
        if cond_token_dim > 0:
            cond_embed_dim = cond_token_dim if not project_cond_tokens else config.hidden_size
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0
            
        # 全局条件处理
        if global_cond_dim > 0:
            global_embed_dim = global_cond_dim if not project_global_cond else config.hidden_size
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )
            
        # 前置条件处理
        if prepend_cond_dim > 0:
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, config.hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            )
            
        self.input_concat_dim = input_concat_dim
        dim_in = io_channels + self.input_concat_dim
        self.patch_size = patch_size
        
        # 创建 RWKV 模型
        global_dim = None
        if self.global_cond_type == "adaLN":
            global_dim = config.hidden_size
        
        self.rwkv = ContinuousRWKV(
            config,
            dim_in=dim_in * patch_size,
            dim_out=io_channels * patch_size,
            cross_attend=cond_token_dim > 0,
            cond_token_dim=cond_embed_dim,
            global_cond_dim=global_dim,
            **kwargs
        )
        
        # 预处理和后处理卷积
        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)
        
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def _forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cross_attn_cond: Optional[torch.Tensor] = None,
        cross_attn_cond_mask: Optional[torch.Tensor] = None,
        input_concat_cond: Optional[torch.Tensor] = None,
        global_embed: Optional[torch.Tensor] = None,
        prepend_cond: Optional[torch.Tensor] = None,
        prepend_cond_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # print(f"x: {x.shape}")
        # 处理交叉注意力条件
        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)
            logger.debug(f"cross_attn_cond: {cross_attn_cond.shape}")
        # 处理全局条件
        if global_embed is not None:
            global_embed = self.to_global_embed(global_embed)
            logger.debug(f"global_embed: {global_embed.shape}")
        # 处理前置条件
        prepend_inputs = None
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            prepend_cond = self.to_prepend_embed(prepend_cond)
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask
                
        # 处理输入连接条件
        if input_concat_cond is not None:
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2],), mode='nearest')
            x = torch.cat([x, input_concat_cond], dim=1)
            
        # 处理时间步嵌入
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]))
        logger.debug(f"timestep_embed: {timestep_embed.shape} and timestep_cond_type: {self.timestep_cond_type}")
        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            # 确保时间步嵌入的维度与输入张量的序列长度匹配
            if timestep_embed.shape[-1] != x.shape[2]:
                timestep_embed = F.interpolate(timestep_embed.unsqueeze(1), (x.shape[2],), mode='nearest').squeeze(1)
            x = torch.cat([x, timestep_embed.unsqueeze(1).expand(-1, -1, x.shape[2])], dim=1)
            
        # 处理全局条件
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)
            prepend_length = prepend_inputs.shape[1]
            
        # 预处理
        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")
        
        # RWKV 处理
        extra_args = {}
        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed
        else:
            extra_args["global_cond"] = None
            
        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)
        # print(f'global_embed: {global_embed.shape}')
        # print(f"x: {x.shape} before rwkv")
            
        x = self.rwkv(
            x,
            mask=mask,
            prepend_embeds=prepend_inputs,
            prepend_mask=prepend_mask,
            return_info=return_info,
            context=cross_attn_cond,
            context_mask=cross_attn_cond_mask,
            **extra_args,
            **kwargs
        )
        logger.debug(f"x: {x.shape} after rwkv")
        if return_info:
            x, info = x
            
        # 后处理
        x = rearrange(x, "b t c -> b c t")[:, :, prepend_length:]
        logger.debug(f"x: {x.shape} after postprocess")
        if self.patch_size > 1:
            x = rearrange(x, "b (c p) t -> b c (t p)", p=self.patch_size)
            
        x = self.postprocess_conv(x) + x
        logger.debug(f'x dtype: {x.dtype} and x shape: {x.shape}')
        if return_info:
            return x, info
        
        return x
        
    def forward(
        self, 
        x, 
        t, 
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_interval = (0, 1),
        causal=False,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs):
        assert causal == False, "Causal mode is not supported for DiffusionTransformer"

        model_dtype = next(self.parameters()).dtype
        
        x = x.to(model_dtype)

        t = t.to(model_dtype)

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)

        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.to(model_dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        # Early exit bypasses CFG processing
        if exit_layer_ix is not None:
            assert self.transformer_type == "continuous_transformer", "exit_layer_ix is only supported for continuous_transformer"
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond, 
                cross_attn_cond_mask=cross_attn_cond_mask, 
                input_concat_cond=input_concat_cond, 
                global_embed=global_embed, 
                prepend_cond=prepend_cond, 
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
                **kwargs
            )

        # CFG dropout
        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

        if self.diffusion_objective == "v":
            sigma = torch.sin(t * math.pi / 2)
            alpha = torch.cos(t * math.pi / 2)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            sigma = t

        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None) and (cfg_interval[0] <= sigma[0] <= cfg_interval[1]):

            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if input_concat_cond is not None:
                batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            else:
                batch_input_concat_cond = None

            batch_cond = None
            batch_cond_masks = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:

                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if negative_cross_attn_cond is not None:

                    # If there's a negative cross-attention mask, set the masked tokens to the null embed
                    if negative_cross_attn_mask is not None:
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                        negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                    
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
               
            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None:

                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                           
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
         

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None
            
            batch_output = self._forward(
                batch_inputs, 
                batch_timestep, 
                cross_attn_cond=batch_cond, 
                cross_attn_cond_mask=batch_cond_masks, 
                mask = batch_masks, 
                input_concat_cond=batch_input_concat_cond, 
                global_embed = batch_global_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                return_info = return_info,
                **kwargs)

            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)

            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            # CFG Rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
            else:
                output = cfg_output
           
            if return_info:
                info["uncond_output"] = uncond_output
                return output, info

            return output
            
        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond, 
                cross_attn_cond_mask=cross_attn_cond_mask, 
                input_concat_cond=input_concat_cond, 
                global_embed=global_embed, 
                prepend_cond=prepend_cond, 
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                **kwargs
            )