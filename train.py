"""
Предобучение Orpheus-Zero — кастомный training loop без HF Trainer.

ВАЖНО про эту версию:
  - LR больше НЕ ресетится к пику при каждом перезапуске скрипта.
  - Optimizer и scheduler сохраняются и загружаются целиком.
  - total_steps считается один раз на NUM_EPOCHS вперёд.
  - Если датасет вырос между запусками (total_steps изменился
    относительно того, что было сохранено в прошлый раз) — scheduler
    пересобирается под новую длину расписания и вручную прокручивается
    до текущего шага. Optimizer state (Adam moving averages) при этом
    загружается как обычно — пересчитывать его не нужно.
  - ПЕРЕХОДНЫЙ СЛУЧАЙ: если train_state.json ещё старого формата (нет
    поля total_steps, как у чекпоинта step_17000) — считаем, что
    расписание могло разъехаться, и форсируем пересборку scheduler'а
    на этот единственный запуск. Дальше total_steps уже будет сохраняться,
    и автоматическое сравнение заработает штатно.
  - ПРИ РАСШИРЕНИИ ДАТАСЕТА batch_index тоже сбрасывается в 0 (см. ниже):
    старый batch_index указывал на позицию в СТАРОМ (меньшем) dataloader'е,
    и после мерджа/shuffle это уже не то место, где реально остановились.
    Скипать по нему в новом датасете бессмысленно. step, веса, optimizer
    и scheduler при этом не трогаются - сбрасывается только позиция
    внутри текущей эпохи.

Запуск:
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python train.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import datetime
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from bitsandbytes.optim import AdamW8bit
from model.config import OrpheaConfig
from model.model import OrpheaForCausalLM
from safetensors.torch import load_file
from tqdm import tqdm

# ─── Логирование в файл + консоль одновременно ───────────────────────────────

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

_log_fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)

_original_print = print

def log(*args, **kwargs):
    _original_print(*args, **kwargs)
    _original_print(*args, **kwargs, file=_log_fh)

print = log

print(f"[Лог этого запуска пишется в: {LOG_FILE}]")

TOKENIZER_PATH  = "/home/mixlygames/PycharmProjects/Orpheus_Zero/Extra/orphea_tokenizer"
DATASET_PATH    = "/home/mixlygames/PycharmProjects/Orpheus_Zero/Extra/dataset/tokenized/full"
OUTPUT_DIR      = "/home/mixlygames/PycharmProjects/Orpheus_Zero/checkpoints"
FINAL_MODEL_DIR = "orpheus_zero_v1"
STATE_FILE      = os.path.join(OUTPUT_DIR, "train_state.json")

CONTEXT_LENGTH  = 4096
BATCH_SIZE      = 1
GRAD_ACCUM      = 16
LR              = 5e-5
WARMUP_STEPS    = 100
SAVE_STEPS      = 500
LOG_STEPS       = 10
MAX_GRAD_NORM   = 1.0

NUM_EPOCHS = 4

COLD_START_STEPS  = 300
COLD_START_MIN_LR = 5e-6

VAL_FRACTION   = 0.01   # доля датасета под валидацию (1%)
VAL_STEPS      = 500    # как часто считать eval loss (в глобальных шагах)
VAL_BATCHES    = 50     # сколько батчей валидации гонять за один замер (не весь val-сет, для скорости)

print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.model_max_length = CONTEXT_LENGTH
pad_id = tokenizer.pad_token_id or 0
print(f"Vocab size: {len(tokenizer)}")

print(f"\nЗагрузка датасета из {DATASET_PATH}...")
full_dataset = load_from_disk(DATASET_PATH)
print(f"Чанков всего: {len(full_dataset):,}")

# ─── Train/val split ──────────────────────────────────────────────────────
# Фиксированный seed - чтобы при рестартах val-сет всегда был тем же самым
# набором индексов и не подмешивался в train между запусками.
split = full_dataset.train_test_split(test_size=VAL_FRACTION, seed=42)
dataset = split["train"]
val_dataset = split["test"]
print(f"Train чанков: {len(dataset):,} | Val чанков: {len(val_dataset):,}")

def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"][:CONTEXT_LENGTH], dtype=torch.long) for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    labels = input_ids.clone()
    labels[labels == pad_id] = -100
    return {"input_ids": input_ids, "labels": labels}

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

print("\nИнициализация модели...")
config = OrpheaConfig(
    vocab_size=len(tokenizer),
    hidden_size=1024,
    num_hidden_layers=18,
    num_attention_heads=16,
    num_key_value_heads=8,
    intermediate_size=2816,
    max_position_embeddings=CONTEXT_LENGTH,
    pad_token_id=pad_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_4bit=False,
)

model = OrpheaForCausalLM(config).to(torch.bfloat16).cuda()
model.gradient_checkpointing_enable()
model.train()

total_params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {total_params / 1e6:.1f}M")

steps_per_epoch = len(dataloader) // GRAD_ACCUM
total_steps = steps_per_epoch * NUM_EPOCHS

optimizer = AdamW8bit(model.parameters(), lr=LR, weight_decay=0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nЗапуск предобучения...")
print(f"  Шагов на эпоху:      {steps_per_epoch}")
print(f"  Эпох всего:          {NUM_EPOCHS}")
print(f"  Шагов всего:         {total_steps}")
print(f"  Эффективный батч:    {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Контекст:            {CONTEXT_LENGTH}")

# ─── Автоопределение последнего чекпоинта и восстановление состояния ─────────

def find_latest_checkpoint():
    if not os.path.exists(OUTPUT_DIR):
        return None, 0
    steps = []
    for d in os.listdir(OUTPUT_DIR):
        if d.startswith("step_"):
            try:
                steps.append(int(d.split("_")[1]))
            except ValueError:
                continue
    if not steps:
        return None, 0
    latest = max(steps)
    return os.path.join(OUTPUT_DIR, f"step_{latest}"), latest


latest_ckpt_path, resumed_global_step = find_latest_checkpoint()

start_batch_index = 0
epoch_start = 0
cold_start = False
cold_start_step = 0

if latest_ckpt_path is not None:
    print(f"\nНайден чекпоинт: {latest_ckpt_path}")

    state = load_file(os.path.join(latest_ckpt_path, "model.safetensors"))
    model.load_state_dict(state)
    print("  Веса модели загружены.")

    opt_path = os.path.join(latest_ckpt_path, "optimizer.pt")
    sched_path = os.path.join(latest_ckpt_path, "scheduler.pt")

    previous_total_steps = None
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            saved_state = json.load(f)
            start_batch_index = saved_state.get("batch_index", 0)
            epoch_start = saved_state.get("epoch", 0)
            previous_total_steps = saved_state.get("total_steps", None)
    else:
        start_batch_index = resumed_global_step * GRAD_ACCUM
        epoch_start = start_batch_index // len(dataloader)

    if os.path.exists(opt_path) and os.path.exists(sched_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location="cuda"))
        print("  Состояние optimizer загружено (moving averages сохранены).")

        dataset_extended = (
            previous_total_steps is not None and previous_total_steps != total_steps
        )

        if previous_total_steps is None:
            print("  [!] В train_state.json нет total_steps (старый формат файла, "
                  "как у чекпоинта до расширения датасета).")
            print(f"      Текущий total_steps={total_steps}. Раз датасет расширяется "
                  f"этим запуском — форсирую пересборку scheduler'а, а не гружу "
                  f"scheduler.pt как есть.")
            dataset_extended = True

        if dataset_extended:
            print(f"  [!] total_steps: было {previous_total_steps}, стало {total_steps}.")
            print("      Пересобираю scheduler под новую длину расписания и "
                  "прокручиваю до текущего шага (optimizer state не трогаю).")
            for _ in range(resumed_global_step):
                scheduler.step()
            print(f"      Scheduler прокручен на {resumed_global_step} шагов "
                  f"по новой кривой (total_steps={total_steps}).")

            # Датасет изменился (другой состав/порядок после shuffle) -
            # старый batch_index указывает на батч в СТАРОМ dataloader'е,
            # это уже не то место, где мы реально остановились. Скипать
            # по этому индексу в новом датасете бессмысленно и может
            # либо повторить, либо пропустить не те данные. Поэтому просто
            # начинаем текущую эпоху заново (с батча 0) - step/веса/optimizer/
            # scheduler при этом не трогаем, они остаются как были.
            print("      Датасет расширился -> batch_index сбрасываю в 0 "
                  "(эпоха начнётся заново на новом составе данных, "
                  "step и веса не сбрасываются).")
            start_batch_index = 0
        else:
            scheduler.load_state_dict(torch.load(sched_path))
            print("  Состояние scheduler загружено как есть (total_steps не менялся).")
    else:
        print("  [!] Optimizer/scheduler state не найден — прокручиваю scheduler вручную.")
        print("      Это нормально только один раз, при переходе со старой схемы.")
        for _ in range(resumed_global_step):
            scheduler.step()

        cold_start = True
        cold_start_step = 0
        print("  [!] Включён локальный cold-start warmup на 300 шагов "
              "(LR временно занижен, пока optimizer state 'разгоняется').")

    print(f"  Глобальный шаг: {resumed_global_step} | Эпоха: {epoch_start} | Батч в эпохе: {start_batch_index % len(dataloader)}")
else:
    print("\nЧекпоинтов не найдено — начинаем обучение с нуля.")

step = resumed_global_step
accum_loss = 0.0
optimizer.zero_grad()

GEN_REP_PENALTY = 1.2

TEST_PROMPTS = [
    "Lobotomy Corporation — это",
    "Once upon a time, in a small village",
    "The weather today is quite",
    "Жила-была в одной деревне девочка, которая",
    "Как твои дела? Я сегодня",
]
gen_prompt_index = 0

def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty == 1.0:
        return logits
    unique_ids = torch.unique(generated_ids)
    scores = logits[0, unique_ids]
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits[0, unique_ids] = scores
    return logits


@torch.no_grad()
def run_validation():
    """Считаем средний loss на отложенном val-сете, без backward.
    Гоняем не весь val-сет (может быть долго), а VAL_BATCHES штук -
    для отслеживания тренда этого достаточно."""
    model.eval()
    losses = []
    for j, val_batch in enumerate(val_dataloader):
        if j >= VAL_BATCHES:
            break
        input_ids = val_batch["input_ids"].cuda()
        labels = val_batch["labels"].cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels)
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")

for epoch in range(epoch_start, NUM_EPOCHS):
    print(f"\n{'='*50}\nЭПОХА {epoch + 1}/{NUM_EPOCHS}\n{'='*50}")

    skip_batches = start_batch_index if epoch == epoch_start else 0

    for i, batch in enumerate(tqdm(dataloader, desc=f"Эпоха {epoch + 1}")):
        if i < skip_batches:
            continue

        input_ids = batch["input_ids"].cuda()
        labels    = batch["labels"].cuda()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        accum_loss += loss.item()

        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            if cold_start and cold_start_step < COLD_START_STEPS:
                target_lr = scheduler.get_last_lr()[0]
                warmup_frac = (cold_start_step + 1) / COLD_START_STEPS
                effective_lr = COLD_START_MIN_LR + (target_lr - COLD_START_MIN_LR) * warmup_frac
                for group in optimizer.param_groups:
                    group["lr"] = effective_lr
                cold_start_step += 1
                if cold_start_step >= COLD_START_STEPS:
                    print("  [i] Cold-start warmup завершён, LR передан обратно scheduler'у.")
            else:
                cold_start = False

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 50 == 0:
                model.eval()
                with torch.no_grad():
                    prompt = TEST_PROMPTS[gen_prompt_index % len(TEST_PROMPTS)]
                    gen_prompt_index += 1
                    inp = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).cuda()
                    out_ids = inp.clone()
                    past = None
                    for _ in range(80):
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            o = model(input_ids=out_ids[:, -1:] if past else out_ids, past_key_values=past, use_cache=True)
                        logits = o.logits[:, -1, :].clone()
                        logits = apply_repetition_penalty(logits, out_ids, GEN_REP_PENALTY)
                        logits = logits / 0.8
                        probs = torch.softmax(logits, dim=-1)
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cumsum = torch.cumsum(sorted_probs, dim=-1)
                        sorted_probs[cumsum - sorted_probs > 0.9] = 0
                        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                        probs = probs / probs.sum()
                        next_tok = torch.multinomial(probs, 1)
                        past = o.past_key_values
                        out_ids = torch.cat([out_ids, next_tok], dim=-1)
                        if next_tok.item() == tokenizer.eos_token_id:
                            break
                    print(f"  [GEN prompt='{prompt}'] {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")
                model.train()

            if step % LOG_STEPS == 0:
                print(f"  epoch {epoch + 1}/{NUM_EPOCHS} | step {step}/{total_steps} | loss {accum_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

            if step % VAL_STEPS == 0:
                val_loss = run_validation()
                print(f"  [VAL] step {step}/{total_steps} | val_loss {val_loss:.4f}")

            accum_loss = 0.0

            if step % SAVE_STEPS == 0:
                ckpt = os.path.join(OUTPUT_DIR, f"step_{step}")
                os.makedirs(ckpt, exist_ok=True)
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                torch.save(optimizer.state_dict(), os.path.join(ckpt, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(ckpt, "scheduler.pt"))

                with open(STATE_FILE, "w") as f:
                    json.dump({
                        "epoch": epoch,
                        "batch_index": i + 1,
                        "total_steps": total_steps,
                    }, f)

                print(f"  Сохранено: {ckpt}")

    start_batch_index = 0

print(f"\nСохранение финальной модели в {FINAL_MODEL_DIR}...")
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print("Готово!")