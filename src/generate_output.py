"""
generate_output.py
------------------
MAIN FILE. Runs the whole project from start to end and writes the result files.

    python src/generate_output.py --zip Sample-Raw-No-Punctuations.zip

Pipeline:
    preprocessing.py     -> read + clean the text            (ignore NULL)
    punctuation.py       -> add punctuation, split sentences
    sentence_ordering.py -> find the correct sentence order
    translation.py       -> join 5 sentences each + translate to English
    generate_output.py   -> save everything + print the GPU-memory summary
"""

import argparse
import os
import shutil
import sys

# make "from preprocessing import ..." work no matter where the script is started from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch

from preprocessing import analyse_texts, load_and_clean
from punctuation import punctuate_all
from sentence_ordering import order_all
from translation import translate_all

LIMIT_GB = 1.5          # GPU memory limit from the project rules
memory_log = {}         # peak GPU memory of every step


def report_gpu(step_name):
    """Print the peak GPU memory used since the last report, then reset the counter."""
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        memory_log[step_name] = round(peak, 3)
        status = "OK (under limit)" if peak < LIMIT_GB else "TOO HIGH!"
        print(f"[GPU memory] {step_name}: peak {peak:.2f} GB / limit {LIMIT_GB} GB -> {status}")
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_log[step_name] = 0.0
        print(f"[GPU memory] {step_name}: running on CPU, GPU memory used = 0 GB")


def save_results(results, output_dir):
    """Write the .txt files and one CSV table. Returns the table as a DataFrame."""
    os.makedirs(output_dir, exist_ok=True)
    table_rows = []

    for name, r in results.items():
        # 1) punctuated sentences, still in the shuffled (original file) order
        with open(os.path.join(output_dir, f"{name}_1_punctuated_jumbled.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(r["sentences_jumbled"]))
        # 2) punctuated sentences in the predicted correct order
        with open(os.path.join(output_dir, f"{name}_2_ordered.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(r["sentences_ordered"]))
        # 3) 5-sentence chunks: Hindi + English
        with open(os.path.join(output_dir, f"{name}_3_chunks_hindi_english.txt"), "w", encoding="utf-8") as f:
            for k, (hi, en) in enumerate(zip(r["chunks_hi"], r["chunks_en"]), start=1):
                f.write(f"--- Chunk {k} ---\nHINDI:\n{hi}\n\nENGLISH:\n{en}\n\n")
        for k, (hi, en) in enumerate(zip(r["chunks_hi"], r["chunks_en"]), start=1):
            table_rows.append({"file": name, "chunk_no": k, "hindi": hi, "english": en})

    table = pd.DataFrame(table_rows)
    table.to_csv(os.path.join(output_dir, "all_results.csv"), index=False, encoding="utf-8-sig")
    return table


def main():
    parser = argparse.ArgumentParser(description="Hindi: punctuation -> ordering -> 5-sentence chunks -> English")
    parser.add_argument("--zip", default="Sample-Raw-No-Punctuations.zip", help="ZIP file (or folder) with the .txt files")
    parser.add_argument("--output", default="output", help="folder for the result files")
    parser.add_argument("--chunk-size", type=int, default=5, help="sentences per chunk (task says 5)")
    parser.add_argument("--w-nsp", type=float, default=1.0, help="weight of the BERT next-sentence signal")
    parser.add_argument("--w-emb", type=float, default=1.0, help="weight of the meaning-similarity signal")
    parser.add_argument("--w-tfidf", type=float, default=1.0, help="weight of the shared-words signal")
    parser.add_argument("--fp16", action="store_true", help="use half precision (about half the GPU memory)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- Step 1: read + clean ----
    print("\n=== Step 1: preprocessing ===")
    raw_texts, clean_texts = load_and_clean(args.zip)
    print(analyse_texts(raw_texts, clean_texts).to_string(index=False))

    # ---- Step 2: punctuation ----
    print("\n=== Step 2: punctuation ===")
    jumbled = punctuate_all(clean_texts)
    report_gpu("Step 4 punctuation")

    # ---- Step 3: ordering ----
    print("\n=== Step 3: sentence ordering ===")
    ordered = order_all(jumbled, device, args.w_nsp, args.w_emb, args.w_tfidf, args.fp16, report_gpu)

    # ---- Step 4: chunk + translate ----
    print("\n=== Step 4: chunking + translation ===")
    translated = translate_all(ordered, device, args.chunk_size, args.fp16, report_gpu)

    # ---- Step 5: save ----
    print("\n=== Step 5: saving ===")
    results = {name: {"sentences_jumbled": jumbled[name],
                      "sentences_ordered": ordered[name],
                      **translated[name]} for name in jumbled}
    table = save_results(results, args.output)
    print(table.head(6).to_string(max_colwidth=50))

    print("\nPeak GPU memory per step (limit = %.1f GB):" % LIMIT_GB)
    for step, gb in memory_log.items():
        print(f"  {step:40s} {gb:.2f} GB")

    shutil.make_archive("results", "zip", args.output)
    print(f"\nAll results are in the folder '{args.output}' and in 'results.zip'.")


if __name__ == "__main__":
    main()
