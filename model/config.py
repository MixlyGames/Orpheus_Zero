from transformers import PretrainedConfig


class OrpheaConfig(PretrainedConfig):

    model_type = "orphea"

    def __init__(
        self,
        # Основные параметры
        vocab_size: int = 32000,
        hidden_size: int = 1024,            # было 1536
        num_hidden_layers: int = 18,        # было 24
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,       # GQA: меньше KV голов = меньше памяти
        intermediate_size: int = 2816,      # было 4096
        max_position_embeddings: int = 4096, # было 16384; трейнер использует CONTEXT_LENGTH=4096
        # RoPE theta увеличиваем для длинного контекста (как в LLaMA-3)
        rope_theta: float = 500000.0,

        # Нормализация
        rms_norm_eps: float = 1e-5,

        # Активация
        hidden_act: str = "silu",           # SwiGLU использует silu

        # Dropout
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,

        # Специальные токены
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,

        # Инициализация весов
        initializer_range: float = 0.02,

        # Прочее
        tie_word_embeddings: bool = False,  # не делим веса эмбеддинга и lm_head
        use_cache: bool = True,             # KV cache для инференса
        use_4bit: bool = False,             # 4bit квантизация линейных слоёв (для обучения на слабом железе)

        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.use_4bit = use_4bit

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Быстрая проверка числа параметров
def estimate_params(config: OrpheaConfig) -> int:
    H = config.hidden_size
    L = config.num_hidden_layers
    V = config.vocab_size
    I = config.intermediate_size
    A = config.num_attention_heads
    KV = config.num_key_value_heads
    head_dim = H // A

    # Эмбеддинги
    embed = V * H

    # Один слой трансформера
    # Attention: Q, K, V, O проекции
    q_proj = H * H
    k_proj = H * (KV * head_dim)
    v_proj = H * (KV * head_dim)
    o_proj = H * H
    attn = q_proj + k_proj + v_proj + o_proj

    # FFN (SwiGLU): gate, up, down проекции
    ffn = H * I + H * I + I * H

    # RMSNorm (без bias, только scale)
    norm = H * 2  # pre-attn + pre-ffn

    layer = attn + ffn + norm

    # LM head
    lm_head = H * V if not config.tie_word_embeddings else 0

    # Final norm
    final_norm = H

    total = embed + L * layer + lm_head + final_norm
    return total


if __name__ == "__main__":
    config = OrpheaConfig()
    params = estimate_params(config)
    print(f"Orpheus-Zero конфигурация:")
    print(f"  hidden_size:          {config.hidden_size}")
    print(f"  num_hidden_layers:    {config.num_hidden_layers}")
    print(f"  num_attention_heads:  {config.num_attention_heads}")
    print(f"  num_key_value_heads:  {config.num_key_value_heads}")
    print(f"  intermediate_size:    {config.intermediate_size}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  vocab_size:           {config.vocab_size}")
    print(f"\n  Примерное кол-во параметров: {params / 1e6:.1f}M")