<div align="center">

# Orpheus-Zero

**Пайплайн для обучения LLM с нуля — свой токенизатор, своя очистка датасета, свой training loop.**

![Python](https://img.shields.io/badge/Python-3.10+-3572A5?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-bfloat16-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗-Transformers-FFD21E?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Orpheus--Zero-FFD21E?style=flat-square)](https://huggingface.co/MixlyGames/Orpheus_Zero-base-0.3b)

</div>

---

## Что это

Пайплайн для обучения GPT-style декодера с нуля: свой токенизатор, своя очистка данных, свой training loop без `Trainer` из `transformers`. Никакого чёрного ящика — каждый этап это обычный скрипт, который можно прочитать и понять, что он делает.

Сейчас это базовая модель на ~270M параметров (`Orpheus-Zero`), обученная в основном на русском игровом лоре / фантастике / текстах песен. В перспективе — мультимодалка (текст + аудио + картинки + видео), но это будущее, не то, что тут сейчас лежит.

---

## Пайплайн

Три шага, каждый — отдельный скрипт, гоняются по очереди:

1. **`train_tokenizer.py`** — обучает BPE-токенизатор на текстовом корпусе
2. **`prepare_dataset.py`** — чистит, дедуплицирует и токенизирует корпус
3. **`train.py`** — тренирует модель на готовом датасете

```
текстовый корпус
     │
     ├─ 1. train_tokenizer.py → токенизатор
     │
     ├─ 2. prepare_dataset.py → готовый датасет (чанки токенов)
     │
     └─ 3. train.py → чекпоинты модели
```

---

## Фичи

- **Умный отбор файлов под токенизатор** — вместо того, чтобы скармливать весь корпус, сначала находит файлы, которые быстрее всего покрывают весь алфавит, а остальной объём добирает случайно. Не нужно перемалывать гигабайты текста ради полного алфавита.
- **Дешёвый языковой фильтр** — без скачивания моделей fasttext/cld3, просто доля кириллицы/латиницы/цифр/пунктуации в тексте. Быстро и достаточно, чтобы отсеять китайский/арабский/мусор.
- **Дедупликация, которая не тормозит на больших корпусах** — точный hash-дедуп + SimHash/LSH для почти-дублей, без перебора всех пар файлов.
- **Фильтр качества** — режет слишком короткие, нечитаемые (мало букв) или шаблонно-спамные (повторяющиеся строки) документы ещё до токенизации.
- **Обучение с докидыванием чекпоинтов** — сохраняются веса, optimizer и scheduler. Если между запусками датасет вырос — LR-расписание пересобирается и прокручивается до нужного шага, а не сбрасывается в ноль.
- **Cold-start warmup** — если optimizer state потерян (например, при переезде со старого чекпоинта), LR временно занижается, пока Adam «разгоняется» заново, вместо того чтобы улететь в NaN.
- **Живой контроль качества во время обучения** — периодически считается eval loss и печатаются сэмплы генерации, чтобы видеть на глаз, учится модель или нет, а не только смотреть на циферку лосса.

---

## Стек

| Компонент | Технология |
|---|---|
| Токенизатор | 🤗 `tokenizers` (BPE, byte-level fallback) |
| Модель | свой GPT-style декодер — GQA + RoPE + SwiGLU + RMSNorm |
| Датасет | 🤗 `datasets` |
| Оптимизатор | AdamW 8-bit (`bitsandbytes`) |
| Training loop | чистый PyTorch, bfloat16, gradient checkpointing |

---

## Архитектура (`model/`)

```
model/
├── config.py   # OrpheaConfig — гиперпараметры + оценка числа параметров
└── model.py    # сама модель: attention, MLP, decoder layer, CausalLM-обвязка
```

Написана как обычная `transformers`-модель — наследуется от `PreTrainedModel`, регистрируется в `AutoConfig`/`AutoModelForCausalLM`, так что грузится через стандартный `from_pretrained` и совместима с Unsloth.

Из интересного внутри:

- **GQA** — KV-голов меньше, чем Q-голов (16 / 8), шарятся между группами. Меньше памяти на KV-cache при генерации.
- **flash-attn — опционально.** Если пакет установлен и GPU не старше Ampere (compute capability ≥ 8.0) — модель сама переключается на `flash_attn_func`. Если нет — тихо откатывается на обычный `F.scaled_dot_product_attention`, ничего руками включать не нужно.
- **4-bit линейные слои — опционально.** Через `bitsandbytes` (`Linear4bit`, nf4), включается флагом `use_4bit` в конфиге — для дообучения на слабом железе. Если `bitsandbytes` не стоит, тихо падает обратно на обычный `nn.Linear`.
- **RoPE theta = 500 000** — как в LLaMA-3, под расчёт на более длинный контекст в будущем.
- **`estimate_params()`** в `config.py` — быстро прикинуть число параметров по конфигу, не поднимая саму модель.

---

## Установка

```bash
git clone <repo-url>
cd orpheus-zero
pip install torch transformers tokenizers datasets bitsandbytes safetensors tqdm
```

Нужна GPU с CUDA (обучалось на ноутбучной RTX 4060 8GB). Драйвера и CUDA toolkit — на твоей совести, тут никаких предположений про твою систему не делается.

### 1. Обучить токенизатор

```bash
python train_tokenizer.py --dir ./text --output orphea_tokenizer --vocab-size 32000 --max-gb 3.0
```

### 2. Почистить и токенизировать датасет

```bash
python prepare_dataset.py --dir ./text --tokenizer ./orphea_tokenizer --output dataset/tokenized --context-length 4096
```

### 3. Запустить обучение

Пути и гиперпараметры (датасет, токенизатор, батч, число эпох) правятся константами в начале `train.py`, потом:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py
```

При наличии чекпоинтов в `checkpoints/` обучение само продолжится с последнего.

---

## Заметки

- У всех скриптов есть `--help` с полным списком флагов, команды выше — минимальный набор.
- `train.py` пока хранит пути константами вверху файла, а не через argparse — так и задумано, редактируется руками перед запуском.
- Архитектура модели лежит в `model/` (`config.py`, `model.py`, `__init__.py` с реэкспортом `OrpheaConfig`/`OrpheaForCausalLM`).

---

<div align="center">
<sub>OwlNestTeam</sub>
</div>
