import glob
import os
import re
import zipfile

import pandas as pd


def extract_zip(zip_path, extract_dir="data"):
    if os.path.isdir(zip_path):
        return zip_path
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Could not find '{zip_path}'. Check the path and try again.")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    return extract_dir


def _file_number(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def find_text_files(folder):
    paths = [p for p in glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
             if "__MACOSX" not in p and not os.path.basename(p).startswith("._")]
    if not paths:
        raise FileNotFoundError(f"No .txt files found inside '{folder}'.")
    return sorted(paths, key=_file_number)


def read_text_files(paths):
    raw_texts = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8-sig") as f:
            raw_texts[name] = f.read()
    return raw_texts


def clean_text(text):
    for ch in ["\u200d", "\u200c", "\ufeff"]:
        text = text.replace(ch, "")
    text = text.replace("NULL", " ")
    return " ".join(text.split())


def analyse_texts(raw_texts, clean_texts):
    rows = []
    for name in raw_texts:
        raw, cln = raw_texts[name], clean_texts[name]
        words = cln.split()
        rows.append({
            "file": name,
            "characters (raw)": len(raw),
            "NULL tokens": raw.count("NULL"),
            "words (clean)": len(words),
            "unique words": len(set(words)),
            "English words": len(re.findall(r"[A-Za-z]{2,}", cln)),
            "numbers": len(re.findall(r"\b\d+\b", cln)),
        })
    return pd.DataFrame(rows)


def load_and_clean(zip_path, extract_dir="data"):
    folder = extract_zip(zip_path, extract_dir)
    paths = find_text_files(folder)
    raw_texts = read_text_files(paths)
    clean_texts = {name: clean_text(t) for name, t in raw_texts.items()}
    return raw_texts, clean_texts
