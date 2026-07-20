import os
import glob
import random
import argparse
import json
import heapq
from collections import defaultdict

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


# ============================== КОНФИГ ==============================

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|sep|>",
    "<|image_start|>",
    "<|image_end|>",
    "<|image_pad|>",
    "<|video_start|>",
    "<|video_end|>",
    "<|video_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
]

SAMPLE_BYTES_FOR_CHARSET = 50_000   # сколько байт с файла читать для анализа алфавита
CHUNK_SIZE = 10_000                  # размер чанка при потоковом чтении текста
LANG_SAMPLE_CHARS = 20_000           # сколько символов читать для проверки языка


# ======================= 0. ЯЗЫКОВОЙ ФИЛЬТР =============================
# Без fasttext/pycld3 (лишние зависимости, скачивание моделей) - простая
# эвристика по доле кириллицы+латиницы+цифр+пунктуации в тексте. Для
# отсева CJK/арабского/прочего мусора этого достаточно и это быстро.

import string as _string

_ALLOWED_PUNCT = set(_string.punctuation) | {" ", "\n", "\t", "—", "«", "»", "…", "№"}


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
    if total == 0:
        return False
    return (allowed / total) >= threshold


def filter_files_by_language(files, threshold=0.85):
    """Отсеиваем файлы, где доля ru/en символов ниже порога - экономим бюджет
    объёма и не раздуваем алфавит символами языков, которые модели не нужны."""
    print(f"[1.5/4] Языковой фильтр (порог {threshold:.0%} ru/en символов)...")
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
            print(f"        проверено {i + 1}/{len(files)}, оставлено {len(kept)}")
    print(f"        Прошло фильтр: {len(kept)}/{len(files)} файлов "
          f"({len(kept)/len(files)*100:.1f}%)")
    return kept


# ======================= 1. ОТБОР ФАЙЛОВ =============================

def get_all_files(root_dir):
    files = glob.glob(os.path.join(root_dir, "**", "*.txt"), recursive=True)
    print(f"[1/4] Найдено файлов: {len(files)}")
    if not files:
        raise SystemExit("Файлы .txt не найдены — проверь путь --dir")
    return files


def sample_charset(path, sample_bytes=SAMPLE_BYTES_FOR_CHARSET):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read(sample_bytes)
        return set(clean_text(text))
    except Exception:
        return set()


def select_files(files, max_gb, min_gain=1, seed=42):
    """
    Жадный отбор: сначала гонимся за максимальным покрытием алфавита,
    когда прирост падает — добираем объём случайными файлами.
    Возвращает список путей и итоговое множество символов (для отчёта).
    """
    random.seed(seed)
    max_bytes = max_gb * 1024 ** 3

    print(f"[2/4] Анализ алфавита {len(files)} файлов (сэмплы по {SAMPLE_BYTES_FOR_CHARSET} байт)...")
    file_charsets = {}
    for i, f in enumerate(files):
        file_charsets[f] = sample_charset(f)
        if (i + 1) % 5000 == 0:
            print(f"       просканировано {i + 1}/{len(files)}")

    # --- Lazy greedy set cover через heap (O(n log n) вместо O(n^2)) ---
    # Идея: держим приоритетную очередь по ВЕРХНЕЙ ОЦЕНКЕ прироста.
    # Когда достаём файл с вершины — прирост мог "устареть" (кто-то уже
    # покрыл эти символы), поэтому пересчитываем ТОЛЬКО его и, если оценка
    # просела, кладём обратно в кучу. Так каждый файл пересчитывается
    # много меньше n раз в среднем, а не n раз на каждой итерации.
    print("       Строим очередь приоритетов...")
    heap = [(-len(file_charsets[f]), f) for f in files]
    heapq.heapify(heap)

    covered = set()
    selected = []
    selected_set = set()
    total_size = 0
    processed = 0

    print("       Жадный отбор (lazy greedy)...")
    while heap and total_size < max_bytes:
        neg_gain, f = heapq.heappop(heap)
        if f in selected_set:
            continue
        real_gain = len(file_charsets[f] - covered)
        # если оценка устарела (кто-то уже покрыл часть символов) - обновляем и кладём обратно
        if real_gain < -neg_gain:
            heapq.heappush(heap, (-real_gain, f))
            continue
        if real_gain < min_gain:
            # алфавит практически исчерпан - переходим в фазу случайного добора объёма
            break
        covered |= file_charsets[f]
        selected.append(f)
        selected_set.add(f)
        total_size += os.path.getsize(f)
        processed += 1
        if processed % 600 == 0:
            print(f"       отобрано {processed} файлов, {total_size / 1024**3:.2f} GB, "
                  f"алфавит: {len(covered)} символов")

    if total_size < max_bytes:
        print(f"       Алфавит покрыт, добираем объём случайными файлами до {max_gb} GB...")
        remaining_files = [f for f in files if f not in selected_set]
        random.shuffle(remaining_files)
        for f in remaining_files:
            if total_size >= max_bytes:
                break
            selected.append(f)
            selected_set.add(f)
            total_size += os.path.getsize(f)

    print(f"       Отобрано файлов: {len(selected)}  ({total_size / 1024**3:.2f} GB)")
    print(f"       Уникальных символов в покрытии: {len(covered)}")
    return selected, covered


# ======================= 2. ПОТОКОВОЕ ЧТЕНИЕ ==========================

import unicodedata

# Категории unicode, которые обычно = мусор для текстового корпуса:
# Cc - control chars, Cf - format chars (invisible), Co - private use,
# Cs - surrogate (битые кодировки/суррогатные пары).
_JUNK_CATEGORIES = {"Cc", "Cf", "Co", "Cs"}


def clean_text(text):
    """Убираем control/format/surrogate символы - главный источник раздутого
    алфавита при парсинге веб-мусора, битых кодировок и zalgo-текста."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in _JUNK_CATEGORIES or ch in ("\n", "\t")
    )


def text_iterator(files, chunk_size=CHUNK_SIZE):
    """Читаем файлы по одному, отдаём кусками — в памяти всегда один чанк."""
    for path in files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            text = clean_text(text)
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]
        except Exception:
            continue


# ======================= 3. ОБУЧЕНИЕ ТОКЕНИЗАТОРА ======================

def train_tokenizer(files, output_dir, vocab_size, min_frequency, limit_alphabet):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[3/4] Обучение BPE-токенизатора")
    print(f"       vocab_size={vocab_size}  min_frequency={min_frequency}  "
          f"limit_alphabet={limit_alphabet}  special_tokens={len(SPECIAL_TOKENS)}")

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = NFC()
    # add_prefix_space=False, но с ByteLevel это ок для мультиязычного текста без
    # искусственного разделения "слово" vs "не начало строки"
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        limit_alphabet=limit_alphabet,
        initial_alphabet=ByteLevel.alphabet(),  # гарантируем все 256 byte-level символов в базе
    )

    tokenizer.train_from_iterator(
        text_iterator(files),
        trainer=trainer,
        # length не передаём: text_iterator отдаёт чанки, а не файлы,
        # поэтому len(files) как total вводит в заблуждение (счётчик
        # "переполняет" заявленный total, выглядит как зависание/бесконечный
        # цикл, хотя на деле процесс просто честно идёт дальше).
    )

    # Пост-процессор: автоматически добавляет <|bos|> ... <|eos|> при encode()
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
        special_tokens=[("<|bos|>", bos_id), ("<|eos|>", eos_id)],
    )

    raw_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(raw_path)
    print(f"       Сохранён {raw_path}")

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=raw_path,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        sep_token="<|sep|>",
        additional_special_tokens=[t for t in SPECIAL_TOKENS if t not in [
            "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|sep|>"
        ]],
    )
    fast_tokenizer.save_pretrained(output_dir)
    print(f"       Сохранён HF-токенизатор в {output_dir}/")

    return fast_tokenizer


# ======================= 4. ВАЛИДАЦИЯ ==========================

def validate_tokenizer(tok, output_dir):
    """
    Проверяем, что токенизатор реально рабочий:
      - encode/decode не теряет символы (round-trip)
      - спецтокены на месте
      - средняя длина токена по разным языкам (грубая метрика качества)
    """
    print(f"[4/4] Валидация токенизатора")

    test_sentences = [
        "Привет, как дела?",
        "Hello, how are you?",
        "Lobotomy Corporation — это игра.",
        "Аномалия уровня ALEPH обнаружена в секторе 7.",
        "1234567890 + смешанный текст 42!",
        "Emoji test 🔥🚀 и кириллица вместе",
        "",  # пустая строка — не должна падать
    ]

    errors = 0
    report = []
    for s in test_sentences:
        encoded = tok.encode(s)
        decoded = tok.decode(encoded, skip_special_tokens=True)
        ok = decoded.strip() == s.strip()
        if not ok:
            errors += 1
        report.append({
            "text": s,
            "n_tokens": len(encoded),
            "decoded": decoded,
            "round_trip_ok": ok,
        })
        status = "OK " if ok else "FAIL"
        print(f"       [{status}] '{s}' -> {len(encoded)} токенов -> '{decoded}'")

    # спецтокены должны быть отдельными id, не разбиваться на подтокены
    print("\n       Проверка спецтокенов:")
    for t in SPECIAL_TOKENS:
        tid = tok.convert_tokens_to_ids(t)
        if t == "<|unk|>":
            # unk-токен по определению имеет id == unk_token_id, это не ошибка
            is_ok = tid is not None
        else:
            is_ok = tid is not None and tid != tok.unk_token_id
        print(f"       {'OK ' if is_ok else 'FAIL'}  {t} -> id={tid}")
        if not is_ok:
            errors += 1

    print(f"\n       Итоговый vocab size: {tok.vocab_size}")
    print(f"       Всего спецтокенов: {len(tok.all_special_tokens)}")

    if errors:
        print(f"\n       !!! Обнаружено проблем: {errors}. Проверь корпус/спецтокены.")
    else:
        print(f"\n       Все проверки пройдены, токенизатор рабочий.")

    with open(os.path.join(output_dir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return errors == 0


# ============================== MAIN ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Полный пайплайн обучения токенизатора Orphea")
    parser.add_argument("--dir", required=True, help="Папка с .txt файлами")
    parser.add_argument("--output", default="orphea_tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-freq", type=int, default=10)
    parser.add_argument("--max-gb", type=float, default=3.0,
                         help="Сколько GB текста реально нужно для 32k словаря (не гони 20GB зря)")
    parser.add_argument("--min-gain", type=int, default=1,
                         help="Порог прироста новых символов для жадного отбора файлов")
    parser.add_argument("--limit-alphabet", type=int, default=600)
    parser.add_argument("--skip-file-selection", action="store_true",
                         help="Не гонять умный отбор, взять все файлы как есть (для маленьких корпусов)")
    parser.add_argument("--no-lang-filter", action="store_true",
                         help="Отключить языковой фильтр (по умолчанию отсеивает не-ru/en файлы)")
    parser.add_argument("--lang-threshold", type=float, default=0.85,
                         help="Порог доли ru/en символов для прохождения языкового фильтра")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        raise SystemExit(f"Папка не найдена: {args.dir}")

    all_files = get_all_files(args.dir)

    if not args.no_lang_filter:
        all_files = filter_files_by_language(all_files, args.lang_threshold)
        if not all_files:
            raise SystemExit("После языкового фильтра не осталось файлов — "
                              "понизь --lang-threshold или используй --no-lang-filter")

    if args.skip_file_selection:
        selected_files = all_files
        print("[2/4] Пропускаем отбор, используем все найденные файлы")
    else:
        selected_files, _ = select_files(all_files, args.max_gb, args.min_gain)

    tok = train_tokenizer(
        selected_files,
        args.output,
        args.vocab_size,
        args.min_freq,
        args.limit_alphabet,
    )

    validate_tokenizer(tok, args.output)

    print(f"\nГотово. Токенизатор лежит в: {args.output}/")