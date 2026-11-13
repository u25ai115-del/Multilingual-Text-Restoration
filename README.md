# Multilingual Text Restoration and Translation

## 1. Project Overview

This project focuses on restoring and organizing unpunctuated and jumbled text.

The system processes raw text through multiple stages:

- Punctuation restoration
- Sentence boundary detection
- Sentence ordering
- Grouping sentences into sets of five
- Translation into English

Instead of depending on a single general-purpose language model, the project uses a combination of pretrained NLP models, linguistic rules and similarity-based techniques.

## 2. Proposed Approach

The project follows a multi-stage NLP pipeline.

### Stage 1: Data Loading and Preprocessing

- Load the seven input text files from the ZIP dataset.
- Read text using UTF-8 encoding.
- Remove unwanted Unicode formatting characters.
- Normalize extra whitespace.
- Calculate basic text statistics, including word count, unique words and NULL occurrences.

### Stage 2: Punctuation Restoration

**Model:** `pcs_47lang` (Punctuators ONNX)

The pretrained punctuation model is used to identify sentence boundaries and restore punctuation.

Additional Hindi-specific linguistic rules are applied to improve sentence endings:

- Question detection using Hindi question words.
- Exclamation detection using common Hindi expressions.
- Command detection using imperative words.
- Hindi full stop (।) assignment for ordinary statements.

A rule-based sentence splitter is also implemented as a fallback if the pretrained model is unavailable.

### Stage 3: Sentence Ordering

Sentence ordering uses a hybrid similarity-based approach with three signals.

**A. BERT Next Sentence Prediction**

Model: `bert-base-multilingual-cased`

The intended approach is to calculate how likely one sentence is to follow another using Next Sentence Prediction (NSP) scores.

The implementation checks whether the NSP classification head is available. If it is unavailable, this signal is skipped.

**B. Multilingual Sentence Embeddings**

Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Each sentence is converted into a numerical embedding.

Cosine similarity between sentence embeddings measures semantic similarity.

**C. TF-IDF Similarity**

Character-level TF-IDF features are extracted using n-grams of length 2–4.

This captures similarities in words and character patterns between sentences.

**Combining the three signals**

The similarity matrices are standardized using Z-score normalization and combined using weighted addition.

Current weights:

- NSP: 1.0
- Embeddings: 1.0
- TF-IDF: 1.0

The ordering algorithm then:

1. Creates candidate sentence sequences using a greedy path-selection method.
2. Tries different starting sentences.
3. Improves the sequences by moving small blocks of sentences.
4. Selects the sequence with the lowest overall transition cost.

This is a similarity-based ordering method rather than a guaranteed recovery of the original sentence order.

### Stage 4: Grouping and Translation

**Translation Model:** `Helsinki-NLP/opus-mt-hi-en`

After sentence ordering:

1. Sentences are divided into groups of five.
2. A Hindi-to-English name glossary protects selected names and locations by replacing them with their English spellings.
3. Each group is translated into English.
4. Hindi and English versions of the groups are stored separately.

### Stage 5: Output Generation

The pipeline generates:

- Punctuated sentences in their original jumbled order.
- Punctuated and ordered sentences.
- Groups of five Hindi sentences with their English translations.
- A combined CSV file containing Hindi and English results.

## 3. Pretrained Models Used

The project uses the following pretrained models:

| Model | Purpose | Model Link |
|---|---|---|
| PCS 47-language (`pcs_47lang`) | Punctuation restoration and sentence boundary detection | [Hugging Face](https://huggingface.co/1-800-BAD-CODE/punct_cap_seg_47_language) |
| BERT Base Multilingual Cased | Next Sentence Prediction scoring (conditional on NSP head availability) | [Hugging Face](https://huggingface.co/google-bert/bert-base-multilingual-cased) |
| Multilingual MiniLM L12 v2 | Sentence embeddings and semantic similarity | [Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| OPUS-MT Hindi to English | Hindi-to-English machine translation | [Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-hi-en) |

### Additional Techniques

- **Rule-based linguistic processing:** Hindi question, exclamation and sentence-ending detection.
- **TF-IDF:** Character-level n-gram similarity (2–4).
- **Cosine Similarity:** Measures semantic similarity between sentence embeddings.
- **Z-score Normalization:** Standardizes similarity scores before combining them.
- **Greedy Path Selection and Local Search:** Used to find a sentence sequence with a lower transition cost.

## 4. Our Trained Model – Demo

Our trained model and related project resources are available here:

**[View / Download Our Trained Model](https://drive.google.com/drive/folders/1gYN6MZKh9mFuAzgBSUmbuCwW0Rk8y64R?usp=drive_link)**


## 5. Libraries and Technologies

- Python
- Google Colab
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Punctuators
- ONNX Runtime
- Scikit-learn
- Pandas
- NumPy

## 6. Hardware and Memory Constraint

The project is designed around the specified GPU memory constraint of less than 1.5 GB.

GPU memory usage is monitored during the different model-processing stages.

The implementation also supports CPU execution when a GPU is unavailable.

Actual peak memory usage must be verified for the target environment.

## 7. Execution

1. Open `main_project.ipynb` in Google Colab.
2. Install the dependencies listed in `requirements.txt`.
3. Upload the input ZIP dataset.
4. Execute the notebook cells in sequence.
5. Check the generated results in the output directory.

## 8. Project Objective

The main objective is to develop a structured NLP pipeline that combines punctuation restoration, sentence relationship analysis, sentence ordering and machine translation to process unstructured multilingual text.
