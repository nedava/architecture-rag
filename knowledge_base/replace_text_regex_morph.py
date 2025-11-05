import json
import re
from pathlib import Path
from pymorphy3 import MorphAnalyzer

INPUT_FILE_DIR = Path('./raw_text_data')
OUTPUT_FILE_DIR = Path('./text_data')
DICT_FILE = "terms_map.json"

# --- Инициализация морфоанализатора ---
morph = MorphAnalyzer()

# --- Загрузка словаря ---
with open(DICT_FILE, "r", encoding="utf-8") as f:
    replacements = json.load(f)

# --- Утилиты ---

def match_case(source: str, target: str) -> str:
    """Сохраняет регистр исходного слова."""
    if source.isupper():
        return target.upper()
    elif source[0].isupper():
        return target.capitalize()
    return target.lower()

def inflect_word(word: str, replacement: str) -> str:
    """Согласует форму replacement с грамматикой исходного слова."""
    try:
        src = morph.parse(word)[0]
        repl = morph.parse(replacement)[0]
        inflected = repl.inflect(src.tag.grammemes)
        return inflected.word if inflected else replacement
    except Exception:
        return replacement

def smart_replace(text: str, pattern: str, replacement: str) -> str:
    """
    Выполняет замену по шаблону (регулярке),
    согласовывая форму и регистр при необходимости.
    """
    regex = re.compile(pattern)

    def repl_func(match):
        src_text = match.group(0)
        repl_text = replacement

        # Если слово русское — попробуем склонять
        if re.fullmatch(r"[А-Яа-яЁё]+", src_text) and " " not in replacement:
            repl_text = inflect_word(src_text, repl_text)

        # Сохраняем регистр
        repl_text = match_case(src_text, repl_text)
        return repl_text

    return regex.sub(repl_func, text)

# --- Основная функция ---
def replace_text(text: str) -> str:
    # Сначала более длинные шаблоны (фразы)
    sorted_patterns = sorted(replacements.keys(), key=len, reverse=True)
    for pattern in sorted_patterns:
        text = smart_replace(text, pattern, replacements[pattern])
    return text


def process_file(input_file_path: Path):
    text = input_file_path.read_text(encoding="utf-8")

    result = replace_text(text)

    output_file_path = OUTPUT_FILE_DIR / input_file_path.name
    Path(output_file_path).write_text(result, encoding="utf-8")
    print(f"✅ Готово! Результат сохранён в {output_file_path}")


if __name__ == "__main__":
    for input_file_path in INPUT_FILE_DIR.rglob('*.txt'):
        if input_file_path.is_file():
            process_file(input_file_path)
