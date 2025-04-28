from __future__ import annotations

import torch
from torch import nn
from rwkvfla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Attention, RWKV7FeedForward
from rwkvfla.layers.rwkv7 import LoRA
from rwkvfla.modules import LayerNorm
from typing import Optional, Tuple
from functools import partial
from rwkvfla.ops.rwkv7.chunk import chunk_rwkv7
from rwkvfla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from torch.nn import functional as F
def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
from einops import rearrange
def convert_to_left_padding(
    context: torch.Tensor,
    context_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将context和mask转换为left padding格式
    
    Args:
        context: shape (b, len_ctx, C)
        context_mask: shape (b, len_ctx), 1表示有效,0表示padding
        
    Returns:
        left_padded_context: shape (b, len_ctx, C)
        left_padded_mask: shape (b, len_ctx)
    """
    batch_size, seq_len, hidden_size = context.shape
    
    # 创建结果张量
    left_padded_context = torch.zeros_like(context)
    left_padded_mask = torch.zeros_like(context_mask)
    
    # 对每个batch进行处理
    for b in range(batch_size):
        # 获取当前batch的mask
        mask = context_mask[b]  # (len_ctx,)
        
        # 找到所有有效位置
        valid_indices = torch.where(mask)[0]  # 获取所有值为1的索引
        
        # 计算有效长度
        valid_len = len(valid_indices)
        
        # 将有效内容移到右侧,保持相对顺序
        left_padded_context[b, -valid_len:] = context[b, valid_indices]
        left_padded_mask[b, -valid_len:] = mask[valid_indices]
    
    return left_padded_context, left_padded_mask
class ContextLengthTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(
        self, 
        context: torch.Tensor, 
        target_length: int,
        context_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            context: shape (b, len_ctx, C)
            target_length: 目标长度
            context_mask: shape (b, len_ctx), 1表示有效,0表示padding,
            attention_mask: shape (b, target_length), 1表示有效,0表示padding
        Returns:
            transformed_context: shape (b, target_length, C)
            transformed_mask: shape (b, target_length)
        """
        if context_mask is not None:
            context, context_mask = convert_to_left_padding(context, context_mask)
        
        batch_size, seq_len, hidden_size = context.shape
        
        # 转置维度以适配interpolate
        x = context.transpose(1, 2)  # (b, C, len_ctx)
        
        # 对每个样本单独进行插值
        if attention_mask is not None:
            # 计算每个样本的实际有效长度
            valid_lengths = attention_mask.sum(dim=-1)  # (b,)
            
            # 创建结果张量
            transformed_context = torch.zeros((batch_size, target_length, hidden_size), 
                                           device=context.device, dtype=context.dtype)
            
            # 对每个样本单独处理
            for b in range(batch_size):
                valid_len = valid_lengths[b].item()
                if valid_len > 0:
                    # 只对有效部分进行插值
                    sample = x[b:b+1]  # (1, C, len_ctx)
                    transformed = F.interpolate(
                        sample,
                        size=valid_len,
                        mode='linear',
                        align_corners=True
                    )
                    # 将结果放入对应位置
                    transformed_context[b, :valid_len] = transformed[0].transpose(0, 1)
        else:
            # 如果没有attention_mask，则使用统一的target_length
            transformed_context = F.interpolate(
                x,
                size=target_length,
                mode='linear',
                align_corners=True
            ).transpose(1, 2)
        
        # 处理mask
        if context_mask is not None:
            # 将mask转换为float类型以进行插值
            mask = context_mask.float().unsqueeze(1)  # (b, 1, len_ctx)
            
            if attention_mask is not None:
                # 对每个样本单独处理mask
                transformed_mask = torch.zeros((batch_size, target_length), 
                                             device=mask.device, dtype=torch.bool)
                for b in range(batch_size):
                    valid_len = valid_lengths[b].item()
                    if valid_len > 0:
                        sample_mask = mask[b:b+1]  # (1, 1, len_ctx)
                        transformed = F.interpolate(
                            sample_mask,
                            size=valid_len,
                            mode='nearest'
                        )
                        transformed_mask[b, :valid_len] = transformed[0, 0].bool()
            else:
                # 如果没有attention_mask，则使用统一的target_length
                transformed_mask = F.interpolate(
                    mask,
                    size=target_length,
                    mode='nearest'
                ).squeeze(1).bool()
            
            return transformed_context, transformed_mask
            
        return transformed_context, None
class RWKV7CrossAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = None,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 16,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        fuse_norm: bool = False,
        value_dim: int = None,
        num_hidden_layers: int = None,
        cond_token_dim: int = None,
        **kwargs
    ) -> RWKV7Attention:
        super().__init__()

        self.mode = mode
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        if head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        elif head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        elif num_heads is not None:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = num_heads
        self.head_v_dim = int(self.value_dim // self.num_heads)

        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.fuse_norm = fuse_norm

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=decay_low_rank_dim, activation='tanh')
        if self.layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        if self.fuse_norm:
            self.g_norm = GroupNorm(
                num_groups=self.num_heads,
                hidden_size=self.value_dim,
                elementwise_affine=elementwise_affine,
                eps=self.head_dim*norm_eps,
                bias=True,
            )
        else:
            self.g_norm = nn.GroupNorm(
                num_groups=self.num_heads,
                num_channels=self.value_dim,
                eps=self.head_dim*norm_eps,
                affine=elementwise_affine
            )
        self.context_transformer = ContextLengthTransformer()
        self.cond_token_dim = cond_token_dim
        if cond_token_dim is not None:
            self.cond_token_proj = nn.Linear(cond_token_dim, hidden_size, bias=False)
        else:
            self.cond_token_proj = nn.Identity()
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return

        # Initialize only when we're processing the RWKV7Attention module itself
        if isinstance(module, RWKV7Attention) and self.layer_idx is not None:
            ratio_0_to_1 = self.layer_idx / (self.num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)  # 1 to ~0

            # Create position-based initialization tensor
            with torch.no_grad():
                ddd = torch.ones(1, 1, self.hidden_size)
                for i in range(self.hidden_size):
                    ddd[0, 0, i] = i / self.hidden_size

                # Initialize x_* parameters directly
                self.x_r.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_r.dtype)
                self.x_w.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_w.dtype)
                self.x_k.data = (1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)).to(self.x_k.dtype)
                self.x_v.data = (1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)).to(self.x_v.dtype)
                self.x_a.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_a.dtype)
                self.x_g.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_g.dtype)
                # Set specific bias values for LoRA modules
                # w0 initialization - complex decay speed
                decay_speed = torch.ones(self.hidden_size)
                for n in range(self.hidden_size):
                    decay_speed[n] = -7 + 5 * (n / (self.hidden_size - 1)) ** (
                        0.85 + 1.0 * ratio_0_to_1**0.5
                    )
            # Initialize k_k, k_a, r_k
            nn.init.constant_(self.k_k, 0.85)
            nn.init.constant_(self.k_a, 1.0)
            nn.init.zeros_(self.r_k)

            self.w_lora.set_bias_value(decay_speed + 0.5)

            # v0 initialization - ones (for non-first layers)
            if self.layer_idx != 0:
                self.v_lora._initialize_weights(self.v_lora)
                self.v_lora.set_bias_value(1.0)

            self.r_proj.weight.data.uniform_(-0.5/(self.hidden_size**0.5), 0.5/(self.hidden_size**0.5))
            self.k_proj.weight.data.uniform_(-0.05/(self.hidden_size**0.5), 0.05/(self.hidden_size**0.5))
            self.v_proj.weight.data.uniform_(-0.5/(self.hidden_size**0.5), 0.5/(self.hidden_size**0.5))
            self.o_proj.weight.data.zero_()

        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        v_first: torch.Tensor = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, seq_len, _ = hidden_states.shape

        if self.training:
            # if training, use chunk mode no matter how short the sequence is
            mode = 'chunk'
        else:
            # launching the triton kernel for just one token will actually be slower
            mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if attention_mask is not None:
            hidden_states = hidden_states.mul(attention_mask[:, -hidden_states.shape[-2]:, None])
        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state['conv_state'].unsqueeze(1)
        else:
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state['conv_state']

        # [batch_size, seq_len, hidden_size]
        delta = shifted - hidden_states

        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(hidden_states, delta, self.x_r, self.x_w,
                                                     self.x_k, self.x_v, self.x_a, self.x_g)
        context, context_mask = self.context_transformer(context, seq_len, context_mask,attention_mask)
        context = self.cond_token_proj(context)
        r = self.r_proj(context)
        # Using bf16 for LoRA computation is numerically safe here because:
        # 1. After sigmoid activation:
        #    - Max absolute error (vs float32): 0.003
        #    - Mean absolute error: 0.0004
        # 2. Subsequent scaling by -0.6065 will further reduce relative error
        #    (error scales linearly with constant multiplication)
        # 3. Final compounded error remains within acceptable bounds for bf16 precision
        # Empirical observation confirms bf16 introduces no practical degradation
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()

        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        if self.fuse_norm:
            kk = l2_norm(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim))
        else:
            kk = F.normalize(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim), dim=-1, p=2.0)

        # Prefer addcmul over expanded form for numerical stability in bf16:
        # 1. Fused Multiply-Add (FMA) in addcmul reduces intermediate rounding:
        #    - Single op vs original 3 ops (mul, sub, mul)
        #    - 1 less intermediate value storage (bf16 write->read overhead)
        # 2. Mathematically equivalent to k*(1 + (a-1)*self.k_a)
        #    but with better precision preservation
        # 3. Particularly crucial for bf16 where intermediate values easily lose precision
        k = k.addcmul(k * (a - 1), self.k_a)

        # dealing with left-padding
        if attention_mask is not None:
            v = v * attention_mask[:, -v.shape[-2]:, None]
        r, w, k, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim), (r, w, k, a))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        rwkv7_fn = chunk_rwkv7 if mode == 'chunk' else fused_recurrent_rwkv7
        cu_seqlens = kwargs.get('cu_seqlens', None)
        o, recurrent_state = rwkv7_fn(
            r=r,
            log_w=w,
            k=k,
            v=v,
            a=-kk,
            b=kk * a,
            scale=1.,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=hidden_states[:, -1],
                layer_idx=self.layer_idx,
                offset=r.shape[1]
            )

        if self.fuse_norm:
            o = self.g_norm(rearrange(o, '... h d -> ... (h d)'))
        else:
            o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(batch_size, seq_len, -1)

        o = o + ((r * k * self.r_k).sum(-1, keepdim=True) * v).view(batch_size, seq_len, -1)
        o = self.o_proj(o * g)

        return o, None, past_key_values, v_first
class RWKV7Block(nn.Module):
    def __init__(
        self,
        config: RWKV7Config,
        layer_idx: int,
        global_cond_dim: int = None,
        cross_attend: bool = False,
        dim_context: Optional[int] = None,
        layer_scale: bool = False,
    ) -> RWKV7Block:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.cross_attend = cross_attend

        if config.norm_first and layer_idx == 0:
            self.pre_norm = LayerNorm(
                config.hidden_size,
                bias=config.norm_bias,
                eps=config.norm_eps
            )
        self.attn_norm = LayerNorm(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps
        )
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                qkv_bias=config.attn['qkv_bias'],
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx
            )
        else:
            self.attn = RWKV7Attention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                decay_low_rank_dim=config.decay_low_rank_dim,
                gate_low_rank_dim=config.gate_low_rank_dim,
                a_low_rank_dim=config.a_low_rank_dim,
                v_low_rank_dim=config.v_low_rank_dim,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
                value_dim=config.value_dim[layer_idx],
                num_hidden_layers=config.num_hidden_layers
            )
        self.ffn_norm = LayerNorm(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps
        )
        self.ffn = RWKV7FeedForward(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_idx=layer_idx,
            num_hidden_layers=config.num_hidden_layers
        )

        self.global_cond_dim = global_cond_dim
        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Parameter(torch.randn(6*config.hidden_size)/config.hidden_size**0.5)

        # 添加交叉注意力相关组件
        if cross_attend:
            self.cross_attend_norm = LayerNorm(
                config.hidden_size,
                bias=config.norm_bias,
                eps=config.norm_eps
            )
            self.cross_attn = RWKV7CrossAttention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                decay_low_rank_dim=config.decay_low_rank_dim,
                gate_low_rank_dim=config.gate_low_rank_dim,
                a_low_rank_dim=config.a_low_rank_dim,
                v_low_rank_dim=config.v_low_rank_dim,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
                value_dim=config.value_dim[layer_idx],
                num_hidden_layers=config.num_hidden_layers,
                cond_token_dim=dim_context
            )
            self.cross_attn_scale = LayerScale(config.hidden_size) if layer_scale else nn.Identity()
        else:
            self.cross_attn_scale = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        v_first: torch.Tensor = None,
        global_cond: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.global_cond_dim is not None and global_cond is not None:
            # 将 global_cond 转换为 scale 和 shift 参数
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)
            
            # 应用 pre_norm 和 scale/shift
            if hasattr(self, 'pre_norm'):
                hidden_states = self.pre_norm(hidden_states)
            hidden_states = hidden_states * (1 + scale_self) + shift_self
            
            # 应用 attention
            hidden_states = self.attn_norm(hidden_states)
            hidden_states, attentions, past_key_values, v_first = self.attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                v_first=v_first,
                **kwargs
            )
            hidden_states = hidden_states * torch.sigmoid(1 - gate_self)
            
            # 应用 feedforward
            hidden_states = self.ffn_norm(hidden_states)
            hidden_states = hidden_states * (1 + scale_ff) + shift_ff
            hidden_states, past_key_values = self.ffn(hidden_states, attention_mask, past_key_values)
            hidden_states = hidden_states * torch.sigmoid(1 - gate_ff)
        else:
            residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states
            hidden_states = self.attn_norm(residual)
            hidden_states, attentions, past_key_values, v_first = self.attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                v_first=v_first,
                **kwargs
            )
            if self.config.fuse_norm:
                hidden_states, residual = self.ffn_norm(hidden_states, residual, True)
            else:
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = self.ffn_norm(hidden_states)
            hidden_states, past_key_values = self.ffn(hidden_states, attention_mask, past_key_values)
            hidden_states = residual + hidden_states

        # 添加交叉注意力处理
        if self.cross_attend and context is not None:
            hidden_states = hidden_states + self.cross_attn_scale(
                self.cross_attn(
                    self.cross_attend_norm(hidden_states),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    v_first=v_first,
                    context=context,
                    context_mask=context_mask,
                    **kwargs
                )[0]
            )

        outputs = (hidden_states, attentions, past_key_values, v_first)
        return outputs

class ContinuousRWKV(nn.Module):
    def __init__(
        self,
        config: RWKV7Config,
        *,
        dim_in = None,
        dim_out = None,
        cross_attend = False,
        cond_token_dim = None,
        final_cross_attn_ix = -1,
        global_cond_dim = None,
        causal = False,
        zero_init_branch_outputs = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.dim = config.hidden_size
        self.depth = config.num_hidden_layers
        self.causal = causal
        self.cross_attend = cross_attend
        self.final_cross_attn_ix = final_cross_attn_ix
        
        # 输入输出投影层
        self.project_in = nn.Linear(dim_in, self.dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(self.dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        # 创建 RWKV 层
        self.layers = nn.ModuleList([
            RWKV7Block(
                config, 
                i, 
                global_cond_dim=global_cond_dim,
                cross_attend=cross_attend and (self.final_cross_attn_ix == -1 or i <= self.final_cross_attn_ix),
                dim_context=cond_token_dim
            ) for i in range(self.depth)
        ])

        # 全局条件处理
        self.global_cond_embedder = None
        if global_cond_dim is not None:
            self.global_cond_embedder = nn.Sequential(
                nn.Linear(global_cond_dim, self.dim),
                nn.SiLU(),
                nn.Linear(self.dim, self.dim * 6)
            )

        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        context = None,  # 添加 context 参数
        context_mask = None,  # 添加 context_mask 参数
        return_info = False,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        # 转换输入数据类型
        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)

        info = {
            "hidden_states": [],
        }

        # 输入投影
        x = self.project_in(x)

        # 处理前置嵌入
        if prepend_embeds is not None:
            batch_size, prepend_max_len, prepend_dim = prepend_embeds.shape
            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'
            
            # 计算每个样本的实际长度
            if prepend_mask is not None:
                prepend_lengths = prepend_mask.sum(dim=-1).long()  # (batch,)
            else:
                prepend_lengths = torch.full((batch_size,), prepend_max_len, device=x.device)
            
            # 计算原始序列的实际长度
            if mask is not None:
                seq_lengths = mask.sum(dim=-1).long()  # (batch,)
            else:
                seq_lengths = torch.full((batch_size,), x.shape[1], device=x.device)
            
            # 计算新的最大长度
            new_max_len = (prepend_lengths + seq_lengths).max().item()
            
            # 创建新的张量
            new_x = torch.zeros((batch_size, new_max_len, prepend_dim), device=x.device, dtype=x.dtype)
            new_mask = torch.zeros((batch_size, new_max_len), device=x.device, dtype=torch.bool)
            
            # 对每个样本进行填充
            for i in range(batch_size):
                prepend_len = prepend_lengths[i]
                seq_len = seq_lengths[i]
                
                # 填充前置嵌入
                if prepend_len > 0:
                    new_x[i, :prepend_len] = prepend_embeds[i, :prepend_len]
                    if prepend_mask is not None:
                        new_mask[i, :prepend_len] = prepend_mask[i, :prepend_len]
                    else:
                        new_mask[i, :prepend_len] = True
                
                # 填充原始序列
                if seq_len > 0:
                    new_x[i, prepend_len:prepend_len+seq_len] = x[i, :seq_len]
                    if mask is not None:
                        new_mask[i, prepend_len:prepend_len+seq_len] = mask[i, :seq_len]
                    else:
                        new_mask[i, prepend_len:prepend_len+seq_len] = True
            
            x = new_x
            mask = new_mask

        # 处理全局条件
        if global_cond is not None and self.global_cond_embedder is not None:
            global_cond = self.global_cond_embedder(global_cond)

        

        # RWKV 层处理
        hidden_states = x
        past_key_values = None
        v_first = None
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, _, past_key_values, v_first = checkpoint(
                    layer,
                    hidden_states,
                    mask,
                    past_key_values,
                    False,  # use_cache
                    False,  # output_attentions
                    v_first,
                    global_cond,
                    context,  # 添加 context
                    context_mask,  # 添加 context_mask
                    **kwargs
                )
            else:
                hidden_states, _, past_key_values, v_first = layer(
                    hidden_states,
                    mask,
                    past_key_values,
                    False,  # use_cache
                    False,  # output_attentions
                    v_first,
                    global_cond,
                    context,  # 添加 context
                    context_mask,  # 添加 context_mask
                    **kwargs
                )
            
            if return_info:
                info["hidden_states"].append(hidden_states)

        # 输出投影
        hidden_states = self.project_out(hidden_states)

        if return_info:
            return hidden_states, info
        
        return hidden_states 