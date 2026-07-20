import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import OrpheaConfig

# bitsandbytes опционально — если нет, используем обычный nn.Linear
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# flash-attn опционально — если нет или GPU старая (compute capability < 8.0),
# тихо откатываемся на обычный SDPA. Ничего в остальном коде менять не нужно.
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
except ImportError:
    HAS_FLASH_ATTN = False


def Linear(in_features, out_features, bias=False, use_4bit=False):
    """Фабрика линейных слоёв — обычный или 4bit в зависимости от флага."""
    if use_4bit and HAS_BNB:
        return bnb.nn.Linear4bit(
            in_features, out_features, bias=bias,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
        )
    return nn.Linear(in_features, out_features, bias=bias)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class OrpheaRMSNorm(nn.Module):
    """RMSNorm — нормализация без вычитания среднего, быстрее LayerNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

class OrpheaRotaryEmbedding(nn.Module):
    """RoPE — относительные позиционные эмбеддинги через вращение."""

    def __init__(self, dim: int, max_position_embeddings: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Attention (GQA)
# ---------------------------------------------------------------------------

class OrpheaAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    num_key_value_heads < num_attention_heads — KV головы шарятся между Q головами.
    Экономит память и ускоряет инференс.

    Если доступен flash-attn (HAS_FLASH_ATTN=True) и нет attention_mask
    (паддинга внутри батча) — используется flash_attn_func, который
    существенно быстрее и экономнее по памяти, чем обычный SDPA.
    Иначе — прежний путь через F.scaled_dot_product_attention.
    """

    def __init__(self, config: OrpheaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, use_4bit=config.use_4bit)
        self.k_proj = Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, use_4bit=config.use_4bit)
        self.v_proj = Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, use_4bit=config.use_4bit)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, use_4bit=config.use_4bit)

        self.rotary_emb = OrpheaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.shape

        # Общая для обоих путей проекция — здесь тензоры ещё в формате
        # (batch, seq, heads, head_dim), transpose делаем только для SDPA-пути.
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # FA2 годится только без attention_mask (без паддинга в батче) и на GPU.
        use_fa2 = HAS_FLASH_ATTN and attention_mask is None and hidden_states.is_cuda

        if use_fa2:
            # flash_attn_func ждёт (batch, seq, heads, head_dim) — rotary применяем
            # с unsqueeze_dim=2 (ось heads стоит на месте 2, а не 1 как в SDPA-пути).
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=1)  # dim=1 — seq в этом формате
                v = torch.cat([past_key_value[1], v], dim=1)
            past_key_value = (k, v) if use_cache else None

            # flash_attn_func умеет GQA нативно — repeat_interleave не нужен,
            # экономит память по сравнению со SDPA-путём.
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
            )  # -> (batch, seq, heads, head_dim)

            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)  # unsqueeze_dim=1 по умолчанию

            # KV cache
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            past_key_value = (k, v) if use_cache else None

            # GQA: expand KV головы до числа Q голов
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None,
            )

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


# ---------------------------------------------------------------------------
# FFN (SwiGLU)
# ---------------------------------------------------------------------------

class OrpheaMLP(nn.Module):
    """
    SwiGLU FFN — gate_proj * silu(up_proj), потом down_proj.
    Лучше обычного ReLU по качеству.
    """

    def __init__(self, config: OrpheaConfig):
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False, use_4bit=config.use_4bit)
        self.up_proj   = Linear(config.hidden_size, config.intermediate_size, bias=False, use_4bit=config.use_4bit)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False, use_4bit=config.use_4bit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class OrpheaDecoderLayer(nn.Module):
    def __init__(self, config: OrpheaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = OrpheaAttention(config, layer_idx)
        self.mlp = OrpheaMLP(config)
        self.input_layernorm = OrpheaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OrpheaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Основная модель
# ---------------------------------------------------------------------------

class OrpheaModel(nn.Module):
    """Backbone без lm_head — нужен чтобы base_model_prefix работал корректно с Unsloth/HF."""

    def __init__(self, config: OrpheaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            OrpheaDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = OrpheaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False


class OrpheaForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Orpheus-Zero — авторегрессивная языковая модель.
    Совместима с HuggingFace Transformers и Unsloth.
    """

    config_class = OrpheaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OrpheaDecoderLayer"]

    def __init__(self, config: OrpheaConfig):
        super().__init__(config)

        self.model = OrpheaModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False, use_4bit=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # -----------------------------------------------------------------
    # Gradient checkpointing — HF-совместимые методы.
    # Проксируют флаг на self.model (backbone), который реально
    # проверяется в forward(). Без этого стандартный
    # model.gradient_checkpointing_enable() из HF/Unsloth не сработал бы,
    # т.к. они выставляют флаг на верхнеуровневой модели, а forward
    # смотрел только на self.model.gradient_checkpointing.
    # -----------------------------------------------------------------
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.model.gradient_checkpointing = False

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,  # noqa: не используется, нужен для совместимости с HF API
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            hidden_states = self.model.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        bsz, seq_len, _ = hidden_states.shape

        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=hidden_states.device).unsqueeze(0)

        causal_mask = None
        if attention_mask is not None:
            causal = torch.full(
                (seq_len, seq_len), torch.finfo(hidden_states.dtype).min,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            causal = torch.triu(causal, diagonal=1)
            causal = causal.unsqueeze(0).unsqueeze(0)

            pad_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) \
                       * torch.finfo(hidden_states.dtype).min

            causal_mask = causal + pad_mask

        new_past_key_values: list = []

        for i, layer in enumerate(self.model.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.model.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states, past_kv = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    None,
                    False,
                    use_reentrant=False,
                )
            else:
                hidden_states, past_kv = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )

            if use_cache:
                new_past_key_values.append(past_kv)

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        out_past = new_past_key_values if use_cache else None

        if not return_dict:
            output = (logits,) + (out_past,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=out_past,
        )


# ---------------------------------------------------------------------------
# Регистрация в HuggingFace AutoModel
# ---------------------------------------------------------------------------

from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("orphea", OrpheaConfig)
AutoModelForCausalLM.register(OrpheaConfig, OrpheaForCausalLM)