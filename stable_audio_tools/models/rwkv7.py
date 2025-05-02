from __future__ import annotations

import torch
from torch import nn
from rwkvfla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from rwkvfla.models.rwkv7.modeling_rwkv7 import RWKV7Attention, RWKV7FeedForward
from rwkvfla.layers.rwkv7 import LoRA
from rwkvfla.modules import LayerNorm
from typing import Optional, Tuple, Literal
from functools import partial, reduce
from rwkvfla.ops.rwkv7.chunk import chunk_rwkv7
from rwkvfla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from torch.nn import functional as F
def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
from einops import rearrange
try:
    from flash_attn import flash_attn_func, flash_attn_kvpacked_func
except ImportError as e:
    print(e)
    print('flash_attn not installed, disabling Flash Attention')
    flash_attn_kvpacked_func = None
    flash_attn_func = None
def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head
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

class TransformerCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm: Literal['l2', 'ln', 'none'] = 'none',
        natten_kernel_size = None,
        sliding_window = [-1, -1],
        feat_scale = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_context if dim_context is not None else dim
        
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        if qk_norm not in ['l2', 'ln', 'none']:
            raise ValueError(f'qk_norm must be one of ["l2", "ln", "none"], got {qk_norm}')
            
        self.qk_norm = qk_norm

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)

        # Using 1d neighborhood attention
        self.natten_kernel_size = natten_kernel_size
        if natten_kernel_size is not None:
            return

        self.use_pt_flash = torch.cuda.is_available() 
        self.use_fa_flash = torch.cuda.is_available() and flash_attn_func is not None

        self.sdp_kwargs = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )

        self.sliding_window = sliding_window
        if not (sliding_window[0] == -1 and sliding_window[1] == -1)  and not self.use_fa_flash:
            print('Sliding window is being used, but Flash Attention is not. Please install Flash Attention to get correct results')

        self.feat_scale = feat_scale

        if self.feat_scale:
            self.lambda_dc = nn.Parameter(torch.zeros(dim))
            self.lambda_hf = nn.Parameter(torch.zeros(dim))

    def flash_attn(
            self,
            q, 
            k, 
            v,
            mask = None,
            causal = None
    ):
        batch, heads, q_len, _, k_len, device = *q.shape, k.shape[-2], q.device
        kv_heads = k.shape[1]
        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])
        # print(f'q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}, heads: {heads}, kv_heads: {kv_heads}')
        if heads != kv_heads:
            # Repeat interleave kv_heads to match q_heads
            heads_per_kv_head = heads // kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))
            # print(f'after repeat interleave: k.shape: {k.shape}, v.shape: {v.shape}')
        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        causal = self.causal if causal is None else causal

        if q_len == 1 and causal:
            causal = False
        
        if mask is not None:
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if mask is None:
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if mask is not None and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim = -1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False
        
        #with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
        # print(f'q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}')
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = mask,
            is_causal = causal
        )

        # for a row that is entirely masked out, should zero out the output of that row token

        if row_is_entirely_masked is not None:
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    def apply_qk_layernorm(self, q, k):
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        causal = None
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h = h)

            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            # print(f'k.shape: {k.shape}, v.shape: {v.shape}')
            # print(f'kv_input.shape: {kv_input.shape}')
            # print(f'q.shape: {q.shape}')
            # print(f'h: {h}, kv_h: {kv_h}')
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
        # Normalize q and k for cosine sim attention
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        elif self.qk_norm == "ln":
            q, k = self.apply_qk_layernorm(q, k)

        if rotary_pos_emb is not None and not has_context:
            freqs, _ = rotary_pos_emb

            q_dtype = q.dtype
            k_dtype = k.dtype

            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)

            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)
        
        input_mask = context_mask 

        if input_mask is None and not has_context:
            input_mask = mask

        # determine masking
        masks = []
        final_attn_mask = None # The mask that will be applied to the attention matrix, taking all masks into account

        if input_mask is not None:
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        # Other masks will be added here later

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if self.natten_kernel_size is not None:
            if natten is None:
                raise ImportError('natten not installed, please install natten to use neighborhood attention')
            
            dtype_in = q.dtype
            q, k, v = map(lambda t: t.to(torch.float32), (q, k, v))

            attn = natten.functional.na1d_qk(q, k, kernel_size = self.natten_kernel_size, dilation=1)

            if final_attn_mask is not None:
                attn = attn.masked_fill(final_attn_mask, -torch.finfo(attn.dtype).max)

            attn = F.softmax(attn, dim=-1, dtype=torch.float32)

            out = natten.functional.na1d_av(attn, v, kernel_size = self.natten_kernel_size, dilation=1).to(dtype_in)

        # Prioritize Flash Attention 2
        elif self.use_fa_flash:
            assert final_attn_mask is None, 'masking not yet supported for Flash Attention 2'
            # Flash Attention 2 requires FP16 inputs
            fa_dtype_in = q.dtype

            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k, v))

            if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
                q, k, v = map(lambda t: t.to(torch.float16), (q, k, v))
            
            out = flash_attn_func(q, k, v, causal = causal, window_size=self.sliding_window)
            
            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')

        # Fall back to PyTorch implementation
        elif self.use_pt_flash:
            out = self.flash_attn(q, k, v, causal = causal, mask = final_attn_mask)

        else:
            # Fall back to custom implementation

            if h != kv_h:
                # Repeat interleave kv_heads to match q_heads
                heads_per_kv_head = h // kv_h
                k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

            scale = 1. / (q.shape[-1] ** 0.5)

            kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

            dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
            
            i, j, dtype = *dots.shape[-2:], dots.dtype

            mask_value = -torch.finfo(dots.dtype).max

            if final_attn_mask is not None:
                dots = dots.masked_fill(~final_attn_mask, mask_value)

            if causal:
                causal_mask = self.create_causal_mask(i, j, device = device)
                dots = dots.masked_fill(causal_mask, mask_value)

            attn = F.softmax(dots, dim=-1, dtype=torch.float32)
            attn = attn.type(dtype)

            out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        # Communicate between heads
        
        # with autocast(enabled = False):
        #     out_dtype = out.dtype
        #     out = out.to(torch.float32)
        #     out = self.to_out(out).to(out_dtype)
        out = self.to_out(out)

        if self.feat_scale:
            out_dc = out.mean(dim=-2, keepdim=True)
            out_hf = out - out_dc

            # Selectively modulate DC and high frequency components
            out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf

        if mask is not None:
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        return out

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
        query_dim: Optional[int] = None,
        **kwargs
    ) -> RWKV7CrossAttention:
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

        if query_dim is None:
            query_dim = hidden_size
        self.r_proj = nn.Linear(query_dim, self.key_dim, bias=False)
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

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module, gain=2 ** -2.5)
        module._is_hf_initialized = True

    def forward(
        self,
        query: torch.Tensor,  # 必须传入 query
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
        query_batch, query_len, query_dim = query.shape
        assert query_len <= seq_len
        assert batch_size == query_batch
        # padding query to the same len with hidden_states
        # Create padding for the left side
        padding_length = seq_len - query_len
        left_padding = torch.zeros(batch_size, padding_length, query_dim, device=query.device, dtype=query.dtype)

        # Concatenate the padding with the query along the sequence length dimension (dim=1)
        padded_query = torch.cat([left_padding, query], dim=1)

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

        # cross_Attention
        r = self.r_proj(padded_query)
        # w (-0.6065, 0)
        # when we apply sigmoid, bf16 input will not have numerical issue
        # (w.float() - w2).abs().max()/mean() = 0.003, 0.0004
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
        cross_attn_mode: Literal['transformer', 'rwkv'] = 'transformer',
    ) -> RWKV7Block:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.cross_attend = cross_attend
        self.cross_attn_mode = cross_attn_mode

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
            if cross_attn_mode == 'transformer':
                self.cross_attn = TransformerCrossAttention(
                    dim=config.hidden_size,
                    dim_context=dim_context,
                    dim_heads=config.head_dim,
                    causal=False,
                    zero_init_output=True,
                    qk_norm='none'
                )
            else:
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
                    query_dim=dim_context
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
        context: Optional[torch.Tensor] = None,  # 添加 context 参数
        context_mask: Optional[torch.Tensor] = None,  # 添加 context_mask 参数
        v_cross_attn: Optional[torch.Tensor] = None,
        cross_past_key_values: Optional[Cache] = None,
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
                v_cross_attn=v_cross_attn,
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
            if self.cross_attn_mode == 'rwkv':
                context, context_mask = convert_to_left_padding(context, context_mask)
                cross_hidden_states,_,cross_past_key_values,v_cross_attn = self.cross_attn(
                    query=context,
                    hidden_states=self.cross_attend_norm(hidden_states),
                    attention_mask=attention_mask,
                    past_key_values=cross_past_key_values,
                    use_cache=False,
                    output_attentions=False,
                    v_first=v_cross_attn,
                    context_mask=context_mask,
                    **kwargs
                )   
            else:  # transformer mode
                cross_hidden_states = self.cross_attn(
                    x=self.cross_attend_norm(hidden_states),
                    context=context,
                    mask=attention_mask,
                    context_mask=context_mask,
                    **kwargs
                )
            hidden_states = hidden_states + self.cross_attn_scale(cross_hidden_states)

        outputs = (hidden_states, attentions, past_key_values, v_first,v_cross_attn,cross_past_key_values)
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
        v_cross_attn = None
        cross_past_key_values = None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, _, past_key_values, v_first,v_cross_attn,cross_past_key_values = checkpoint(
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
                    v_cross_attn,
                    cross_past_key_values,
                    **kwargs
                )
            else:
                hidden_states, _, past_key_values, v_first,v_cross_attn,cross_past_key_values = layer(
                    hidden_states,
                    mask,
                    past_key_values,
                    False,  # use_cache
                    False,  # output_attentions
                    v_first,
                    global_cond,
                    context,  # 添加 context
                    context_mask,  # 添加 context_mask
                    v_cross_attn,
                    cross_past_key_values,
                    **kwargs
                )
            
            if return_info:
                info["hidden_states"].append(hidden_states)

        # 输出投影
        hidden_states = self.project_out(hidden_states)

        if return_info:
            return hidden_states, info
        
        return hidden_states 