# Hindi Text: Punctuation → Sentence Ordering → 5-Sentence Chunks → English Translation

A single Jupyter / Google Colab notebook (`Hindi_Punctuation_Ordering_Translation.ipynb`) that takes raw Hindi text with **no punctuation and shuffled sentences**, restores the punctuation, finds the correct sentence order, joins every 5 sentences and translates them into English.

---

## 1. Problem statement

Input: a ZIP file (`Sample-Raw-No-Punctuations.zip`) with 7 text files (`sample-1.txt` … `sample-7.txt`). Each file contains Hindi sentences that

- have **no punctuation**, and
- are in a **jumbled order**.

Tasks:

1. Identify punctuation marks at the appropriate places.
2. Find the correct order of the sentences.
3. Combine every 5 ordered sentences and translate them into English.

Rules:

- No off-the-shelf LLM such as GPT, Gemini or Claude may be used.
- Every model used must use **less than 1.5 GB of GPU RAM**.
- The word `NULL` in the files is to be ignored.

---

## 2. Understanding the data

| Observation | What we did about it |
|---|---|
| Each file is **one long line** of Hindi words (about 30–70 sentences per file), UTF-8 Devanagari text. | Read with UTF-8 encoding. |
| `NULL` appears where punctuation (comma, dash, quote, full stop, …) was stripped, e.g. "पूजा NULL यज्ञ NULL तथा". | Removed, as instructed. |
| Hidden zero-width characters (`\u200d`, `\u200c`, BOM), e.g. inside *पत्‍नी*. | Removed so the models see normal Hindi spelling. |
| A few English words (e.g. *Bombyx Mori*) and numbers. | Kept as they are. |
| Sentences from different paragraphs/topics are mixed together (for example, sacred cities and the Garhmukteshwar fair in `sample-1`; a movie dialogue in `sample-6`). | Motivated using several ordering signals instead of one. |

The notebook prints a small analysis table (characters, NULL count, words, unique words, English words, numbers per file) and shows examples of the text around `NULL`.

---

## 3. Our approach (overview)

```
 raw .txt files
      │
      ▼
 [Step 3] Clean         remove NULL + hidden characters
      │
      ▼
 [Step 4] Punctuate     pretrained multilingual punctuation model
      │                 → sentences with , । ? !
      ▼
 [Step 5] Order         3 signals per sentence pair → best path through all sentences
      │
      ▼
 [Step 6] Chunk + Translate   groups of 5 sentences → English (names protected)
      │
      ▼
 [Step 7] Save          .txt files + one CSV + results.zip
```

The design principle: **use only small pretrained models, load them one at a time, free the GPU after each, and measure the GPU memory of every step.**

---

## 4. Step-by-step explanation

### Step 0: Install libraries
`transformers`, `sentencepiece`, `sacremoses`, `sentence-transformers`, `punctuators`, `onnxruntime`, `scikit-learn`, `pandas`.

### Step 1: Imports, settings and GPU-memory helper
- Chooses GPU if available, otherwise CPU.
- Settings at the top that can be changed: `CHUNK_SIZE = 5`, signal weights `W_NSP`, `W_EMB`, `W_TFIDF` (all 1.0), `LIMIT_GB = 1.5`.
- Helper functions `free_gpu()` and `report_gpu()` print the **peak GPU memory** after each step.

### Step 2: Load the data
Asks for the ZIP upload (on Colab), extracts it, ignores junk files such as `__MACOSX`, and sorts the files numerically (`sample-1 … sample-7`).

### Step 3: Clean
`clean_text()` removes `NULL` and hidden characters and tidies the spaces.

### Step 4: Punctuation and sentence splitting
- **Model:** `punctuators` (`pcs_47lang`), a small multilingual XLM-RoBERTa model exported to ONNX. It finds the sentence boundaries and adds commas inside sentences. It runs on the CPU, so it uses 0 GB of GPU memory.
- **Question and exclamation marks.** The pretrained model mostly produces only `,` and `।` for Hindi. We therefore added a rule-based `fix_ending()` on top of it:
  - **Question mark `?`:** the sentence contains a question word (क्या, कौन, कब, क्यों, कैसे, कहाँ, कितना, …), but not inside an indirect "… कि …" clause; or it ends with a tag such as "है ना".
  - **Exclamation mark `!`:** the sentence starts with an interjection (अरे, ओह, वाह, हाय, …), or is a short command (जाओ, रुको, चलो, …).
  - Otherwise the sentence ends with the Hindi full stop `।`.
  - If the model itself already produced `?` or `!`, that is kept.
- **Safety net:** if the pretrained punctuation model cannot be downloaded, a simple rule-based splitter is used and a warning is printed.

### Step 5: Finding the order of the sentences
For every pair of sentences (A, B) we compute a score for "B comes right after A" from **three signals**:

| # | Signal | Model | What it captures | GPU memory (measured) |
|---|---|---|---|---|
| 1 | Next-sentence score | `bert-base-multilingual-cased` (Next Sentence Prediction head) | Does B sound like a natural continuation of A? | 0.94 GB |
| 2 | Meaning similarity | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Are A and B about the same topic? | 0.48 GB |
| 3 | Shared words | TF-IDF on character n-grams (no model) | Do A and B mention the same names and words? | 0 |

How the signals are combined:

1. Each score matrix is standardised (z-score, ignoring the diagonal) so they can be added fairly.
2. Total score = `W_NSP·NSP + W_EMB·Embedding + W_TFIDF·TF-IDF`.
3. A search finds the order with the best total score:
   - **Greedy:** start from every sentence and always jump to the best unused next sentence.
   - **Improve:** take the 5 best greedy paths and keep moving blocks of 1–4 sentences to other positions while the total gets better.
4. A check confirms that the "next sentence" part of the BERT checkpoint really loaded. If it is missing, only signals 2 and 3 are used and a warning is printed.

### Step 6: Chunking and translation
- The ordered sentences are cut into groups of **5** (the last group may be shorter).
- **Model:** `Helsinki-NLP/opus-mt-hi-en` (MarianMT, about 75 M parameters, 0.38 GB measured).
- Each Hindi sentence is translated **separately** and the English sentences of a group are then joined into one paragraph. Small translation models sometimes drop sentences when given a long block, so this gives better quality while still producing one English text per 5-sentence group.
- **Keeping names unchanged:** the model changed place and person names (e.g. Varanasi, Allahabad). A `GLOSSARY` dictionary (Hindi name → English spelling) is applied by `protect_names()` **before** translation, so the model leaves the English names alone. More names can be added to `GLOSSARY` at any time.
- Generation is limited (`max_new_tokens=150`, `no_repeat_ngram_size=3`) so the model cannot get stuck repeating itself, and a progress line is printed after every chunk.

### Step 7: Save and download
Creates the `output/` folder and `results.zip` (see section 6).

---

## 5. Compliance with the project rules

| Rule | How it is met |
|---|---|
| No off-the-shelf LLM (GPT, Gemini, Claude, …) | Only small encoder/translation models are used: XLM-RoBERTa-based punctuation model, multilingual BERT, MiniLM sentence encoder, MarianMT. |
| Each model < 1.5 GB GPU RAM | Models are loaded **one at a time** and deleted (`del` + `torch.cuda.empty_cache()`) before the next one. Peak GPU memory per step is printed. |
| Ignore `NULL` | Removed in `clean_text()`. |

**Measured peak GPU memory (Colab GPU):**

| Step | Peak GPU memory |
|---|---|
| Step 4 – punctuation (runs on CPU) | 0.00 GB |
| Step 5a – BERT next-sentence model | 0.94 GB |
| Step 5b – sentence-embedding model | 0.48 GB |
| Step 6 – translation model | 0.38 GB |

The models are **not in memory at the same time**, so the maximum at any one moment is **0.94 GB**, below the 1.5 GB limit.

*Optional:* adding `.half()` after moving each model to the GPU (fp16) roughly halves these numbers, so that even the sum of all steps stays under 1.5 GB.

---

## 6. How to run

1. Open the notebook in **Google Colab**.
2. **Runtime → Change runtime type → GPU (T4).**
3. **Runtime → Run all.** When asked, upload `Sample-Raw-No-Punctuations.zip`.
4. Wait until the last cell prints *"All results are in the folder 'output' and in 'results.zip'"*. The ZIP downloads automatically. It can also be found in the Colab file panel (folder icon on the left) under `/content`.

Colab deletes its files when the session ends, so download `results.zip` before closing the notebook. A full run takes about 5 minutes.

If a cell seems stuck after cancelling, use **Runtime → Restart session** and run all again.

---

## 7. Output files

For each file `sample-N` (N = 1…7):

| File | Content |
|---|---|
| `sample-N_1_punctuated_jumbled.txt` | Punctuated sentences, still in the original shuffled order |
| `sample-N_2_ordered.txt` | Punctuated sentences in the predicted correct order |
| `sample-N_3_chunks_hindi_english.txt` | Every 5 ordered sentences: Hindi text + English translation (**main result**) |

Plus `all_results.csv` with the columns `file, chunk_no, hindi, english` (saved as `utf-8-sig` so Excel shows Hindi correctly).

---

## 8. Changes made after the first version

| Problem noticed | Fix |
|---|---|
| Only `।` and `,` appeared; no `?` or `!` | Rule-based `fix_ending()` added on top of the punctuation model (Step 4). |
| Place and person names were altered in the English translation | `GLOSSARY` + `protect_names()` added before translation (Step 6). |
| Translation cell seemed to hang | Output length capped, repetition blocked, progress printing added; restart-session advice. |
| GPU memory | Confirmed each model is below 1.5 GB and freed before the next one; optional fp16 edits provided. |

---

## 9. Limitations (honest notes)

- **Sentence order is the hardest part.** There is no answer key for the ordering, so the signal weights could not be tuned. Some sentences may land in the wrong place, especially for the mixed-topic and dialogue files (`sample-6`). The weights `W_NSP`, `W_EMB`, `W_TFIDF` can be changed.
- **Question and exclamation marks are rule-based**, so a few sentences may be marked wrongly. The word lists (`QUESTION_WORDS`, `EXCLAIM_START`, `COMMAND_WORDS`) can be extended.
- **Translation quality is limited by the small model** (`opus-mt-hi-en`). Bigger models such as NLLB-600M translate better but do not reliably fit under 1.5 GB.
- **Names** that are not in `GLOSSARY` may still be changed by the translator. Add them to the dictionary. The word प्रेम was deliberately left out because it is both a name and the word for "love".

---

## 10. Libraries and models used

- Python, PyTorch, Hugging Face `transformers`, `sentence-transformers`, `scikit-learn`, `pandas`, `numpy`, `punctuators` + `onnxruntime`.
- Models: `pcs_47lang` (punctuators), `bert-base-multilingual-cased`, `paraphrase-multilingual-MiniLM-L12-v2`, `Helsinki-NLP/opus-mt-hi-en`.


"C:\Users\lucky makwana\Downloads\README.md"
