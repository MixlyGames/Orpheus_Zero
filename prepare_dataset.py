"""
Orphea — подготовка корпуса + токенизация для претрейна.

Пайплайн:
  [1] Языковой фильтр      — отсеиваем не-ru/en файлы (эвристика по алфавиту)
  [2] Очистка текста       — вырезаем control/surrogate/format unicode-мусор + \ufffd
  [3] Фильтр качества      — слишком короткие / нечитаемые / шаблонно-повторные документы
  [4] Точная дедупликация  — побайтовые/почти-побайтовые дубли (hash нормализованного текста)
  [5] Почти-дубликаты      — SimHash + LSH banding (быстро, без O(n^2))
  [6] Токенизация + чанкинг — потоково, без дублирования bos/eos

Использование:
    python prepare_dataset.py \
        --dir /home/mixlygames/PycharmProjects/Orpheus_Zero/text \
        --tokenizer /home/mixlygames/PycharmProjects/Orpheus_Zero/Extra/orphea_tokenizer \
        --output dataset/tokenized \
        --context-length 4096
"""

import os
import re
import glob
import hashlib
import argparse
import unicodedata
import string as _string
from pathlib import Path
from collections import Counter, defaultdict

from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm


# ============================== КОНФИГ ==============================

LANG_SAMPLE_CHARS = 20_000
_ALLOWED_PUNCT = set(_string.punctuation) | {" ", "\n", "\t", "—", "«", "»", "…", "№"}
_JUNK_UNICODE_CATEGORIES = {"Cc", "Cf", "Co", "Cs"}  # control/format/private-use/surrogate


# ======================= [1] ЯЗЫКОВОЙ ФИЛЬТР =============================

def is_target_language(text, threshold=0.85):
    """True, если доля кириллицы+латиницы+цифр+пунктуации в тексте >= threshold."""
    if not text or not text.strip():
        return False
    allowed, total = 0, 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        cp = ord(ch)
        is_cyrillic = 0x0400 <= cp <= 0x04FF
        is_latin = (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A)
        is_digit = ch.isdigit()
        is_punct = ch in _ALLOWED_PUNCT
        if is_cyrillic or is_latin or is_digit or is_punct:
            allowed += 1
    return total > 0 and (allowed / total) >= threshold


def filter_files_by_language(files, threshold=0.85):
    print(f"[1/6] Языковой фильтр (порог {threshold:.0%} ru/en символов)...")
    kept = []
    for i, path in enumerate(files):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                sample = f.read(LANG_SAMPLE_CHARS)
            if is_target_language(sample, threshold):
                kept.append(path)
        except Exception:
            continue
        if (i + 1) % 10000 == 0:
            print(f"       проверено {i + 1}/{len(files)}, оставлено {len(kept)}")
    dropped = len(files) - len(kept)
    print(f"       Прошло: {len(kept)}/{len(files)} ({len(kept)/len(files)*100:.1f}%), "
          f"отсеяно по языку: {dropped}")
    return kept


# ======================= [2] ОЧИСТКА ТЕКСТА =============================

def clean_text(text):
    """
    Убираем:
      - control/format/private-use/surrogate символы (Cc/Cf/Co/Cs)
      - U+FFFD (replacement character) — битые байты из скрейпа/кодировок
    Схлопываем множественные пробелы и тройные переносы строк.
    """
    cleaned = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in _JUNK_UNICODE_CATEGORIES or ch in ("\n", "\t")
    )
    cleaned = cleaned.replace("\ufffd", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ======================= [3] ФИЛЬТР КАЧЕСТВА =============================

def passes_quality(text, min_chars=200, min_alpha_ratio=0.6, max_repeated_line_ratio=0.3):
    """
    Отсеиваем:
      - слишком короткие обрывки (min_chars)
      - мало букв относительно всех непробельных символов
      - шаблонный спам: одна и та же строка повторяется слишком часто
    """
    if len(text) < min_chars:
        return False, "too_short"

    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False, "empty"
    alpha_ratio = sum(c.isalpha() for c in non_space) / len(non_space)
    if alpha_ratio < min_alpha_ratio:
        return False, "low_alpha_ratio"

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 5:
        counts = Counter(lines)
        top_count = counts.most_common(1)[0][1]
        if top_count / len(lines) > max_repeated_line_ratio:
            return False, "repeated_lines"

    return True, "ok"


# ======================= [4] ТОЧНАЯ ДЕДУПЛИКАЦИЯ =============================

def normalized_hash(text):
    """Hash по тексту с нормализованными пробелами и регистром."""
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# ======================= [5] ПОЧТИ-ДУБЛИКАТЫ (SimHash + LSH) =============

def compute_simhash(text, hashbits=64, shingle_size=5):
    """SimHash по shingles из N слов."""
    words = text.split()
    if len(words) < shingle_size:
        shingles = [text] if text else []
    else:
        shingles = [" ".join(words[i:i + shingle_size])
                    for i in range(len(words) - shingle_size + 1)]
    if not shingles:
        return 0

    v = [0] * hashbits
    for shingle in shingles:
        h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16)
        for bit in range(hashbits):
            v[bit] += 1 if (h >> bit) & 1 else -1

    fingerprint = 0
    for bit in range(hashbits):
        if v[bit] > 0:
            fingerprint |= (1 << bit)
    return fingerprint


def hamming_distance(a, b):
    return bin(a ^ b).count("1")


def find_near_duplicates(simhashes, hashbits=64, n_bands=4, max_distance=3):
    """
    LSH banding: делим 64-битный отпечаток на n_bands кусков.
    Документы, совпадающие хотя бы в одном банде, идут в кандидаты.
    Внутри бакетов сравниваем по Хэммингу.
    Возвращает set индексов файлов, которые надо выкинуть.
    """
    band_bits = hashbits // n_bands
    buckets = defaultdict(list)

    for idx, sh in enumerate(simhashes):
        for band in range(n_bands):
            key = (band, (sh >> (band * band_bits)) & ((1 << band_bits) - 1))
            buckets[key].append(idx)

    to_drop = set()
    checked_pairs = set()
    for key, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                pair = (min(a, b), max(a, b))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                if a in to_drop or b in to_drop:
                    continue
                if hamming_distance(simhashes[a], simhashes[b]) <= max_distance:
                    to_drop.add(max(a, b))

    return to_drop


# ======================= ПАЙПЛАЙН ОЧИСТКИ =============================

def clean_and_dedup_files(files, min_chars, min_alpha_ratio, max_repeated_line_ratio,
                           near_dup_distance, skip_near_dup=False):
    print(f"[2-4/6] Очистка + фильтр качества + точная дедупликация "
          f"({len(files)} файлов)...")

    survivors = []
    simhashes = []
    seen_hashes = set()
    stats = Counter()

    for i, path in enumerate(files):
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            stats["read_error"] += 1
            continue

        text = clean_text(raw)
        if not text:
            stats["empty_after_clean"] += 1
            continue

        ok, reason = passes_quality(text, min_chars, min_alpha_ratio, max_repeated_line_ratio)
        if not ok:
            stats[reason] += 1
            continue

        h = normalized_hash(text)
        if h in seen_hashes:
            stats["exact_duplicate"] += 1
            continue
        seen_hashes.add(h)

        survivors.append(path)
        if not skip_near_dup:
            simhashes.append(compute_simhash(text))
        stats["kept"] += 1

        if (i + 1) % 10000 == 0:
            print(f"       обработано {i + 1}/{len(files)}, выжило {len(survivors)}")

    print(f"       После очистки/качества/точного дедупа: {len(survivors)}/{len(files)}")
    for reason, count in stats.most_common():
        if reason != "kept":
            print(f"         отсеяно [{reason}]: {count}")

    if skip_near_dup:
        print("[5/6] Почти-дубликаты: пропущено (--skip-near-dup)")
        return survivors

    print(f"[5/6] Поиск почти-дубликатов (SimHash+LSH, порог Хэмминга {near_dup_distance})...")
    to_drop = find_near_duplicates(simhashes, max_distance=near_dup_distance)
    final_files = [f for idx, f in enumerate(survivors) if idx not in to_drop]
    print(f"       Найдено почти-дубликатов: {len(to_drop)}")
    print(f"       Итоговый чистый корпус: {len(final_files)} файлов")

    return final_files


# ======================= [6] ТОКЕНИЗАЦИЯ + ЧАНКИНГ =============================

def tokenize_and_chunk(file_paths, tokenizer, context_length):
    """
    Читаем файлы, чистим, токенизируем, нарезаем на чанки по context_length.
    post_processor токенизатора добавляет bos/eos автоматически — не дублируем.
    """
    token_buffer = []
    chunks = []

    for path in file_paths:
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        text = clean_text(raw)
        if not text:
            continue

        ids = tokenizer.encode(text, add_special_tokens=True)
        token_buffer.extend(ids)

        cursor = 0
        while len(token_buffer) - cursor >= context_length:
            chunks.append({"input_ids": token_buffer[cursor:cursor + context_length]})
            cursor += context_length

        if cursor > 0:
            token_buffer = token_buffer[cursor:]

    # Остаток — добиваем паддингом если хотя бы полчанка набралось
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if len(token_buffer) >= context_length // 2:
        padded = token_buffer + [pad_id] * (context_length - len(token_buffer))
        chunks.append({"input_ids": padded})

    return chunks


# ============================== MAIN ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка корпуса + токенизация для претрейна Orphea")
    parser.add_argument("--dir", required=True, help="Папка с .txt файлами")
    parser.add_argument("--tokenizer", required=True, help="Путь к обученному токенизатору")
    parser.add_argument("--output", default="dataset/tokenized")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=3000, help="Файлов на один шард")

    parser.add_argument("--lang-threshold", type=float, default=0.85)
    parser.add_argument("--no-lang-filter", action="store_true")

    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.6)
    parser.add_argument("--max-repeated-line-ratio", type=float, default=0.3)

    parser.add_argument("--near-dup-distance", type=int, default=3,
                        help="Порог Хэмминга для SimHash (меньше = строже)")
    parser.add_argument("--skip-near-dup", action="store_true",
                        help="Пропустить поиск почти-дубликатов (быстрее, но менее чисто)")

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        raise SystemExit(f"Папка не найдена: {args.dir}")

    print(f"Сканирование {args.dir}...")
    all_files = sorted(glob.glob(f"{args.dir}/**/*.txt", recursive=True))
    print(f"Найдено файлов: {len(all_files):,}")
    if not all_files:
        raise SystemExit("Файлы .txt не найдены")

    if not args.no_lang_filter:
        all_files = filter_files_by_language(all_files, args.lang_threshold)
        if not all_files:
            raise SystemExit("После языкового фильтра не осталось файлов")

    clean_files = clean_and_dedup_files(
        all_files,
        min_chars=args.min_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        max_repeated_line_ratio=args.max_repeated_line_ratio,
        near_dup_distance=args.near_dup_distance,
        skip_near_dup=args.skip_near_dup,
    )

    if not clean_files:
        raise SystemExit("После всех фильтров не осталось файлов — ослабь пороги")

    print("\nЗагрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = args.context_length
    print(f"Vocab size: {tokenizer.vocab_size}")

    os.makedirs(args.output, exist_ok=True)
    total_chunks = 0
    shard_index = 0

    print(f"\n[6/6] Токенизация (батч = {args.batch_size} файлов)...\n")
    for i in tqdm(range(0, len(clean_files), args.batch_size), desc="Батчи"):
        batch_files = clean_files[i:i + args.batch_size]
        chunks = tokenize_and_chunk(batch_files, tokenizer, args.context_length)

        if not chunks:
            continue

        shard_path = os.path.join(args.output, f"shard_{shard_index:04d}")
        Dataset.from_list(chunks).save_to_disk(shard_path)

        total_chunks += len(chunks)
        shard_index += 1
        del chunks

    print(f"\nОбъединяем {shard_index} шардов...")
    shards = [load_from_disk(os.path.join(args.output, f"shard_{idx:04d}"))
              for idx in range(shard_index)]
    full_dataset = concatenate_datasets(shards)

    final_path = os.path.join(args.output, "full")
    full_dataset.save_to_disk(final_path)

    print(f"\n✅ Готово!")
    print(f"   Исходных файлов:     {len(all_files):,}")
    print(f"   После всех фильтров: {len(clean_files):,}")
    print(f"   Чанков всего:        {len(full_dataset):,}")
    print(f"   Токенов всего:       {len(full_dataset) * args.context_length:,}")
    print(f"   Сохранено в:         {final_path}")