import gc
import re

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

NSP_MODEL_NAME = "bert-base-multilingual-cased"
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_nsp_scores(sentence_lists, device, use_half=False, report=None):
    from transformers import AutoTokenizer, BertForNextSentencePrediction

    tok = AutoTokenizer.from_pretrained(NSP_MODEL_NAME)
    model, info = BertForNextSentencePrediction.from_pretrained(NSP_MODEL_NAME, output_loading_info=True)
    model = model.to(device).eval()
    if use_half and device == "cuda":
        model = model.half()

    ready = not any("seq_relationship" in k for k in info["missing_keys"])
    print("NSP head loaded correctly:", ready)
    scores = {}

    if ready:
        @torch.no_grad()
        def nsp_matrix(sentences):
            n = len(sentences)
            M = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                others = [j for j in range(n) if j != i]
                enc = tok([sentences[i]] * len(others), [sentences[j] for j in others],
                          return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                logits = model(**enc).logits
                log_prob_next = torch.log_softmax(logits.float(), dim=-1)[:, 0]
                M[i, others] = log_prob_next.cpu().numpy()
            return M

        for name, sents in sentence_lists.items():
            scores[name] = nsp_matrix(sents)
            print("NSP done:", name)
    else:
        print("WARNING: NSP part missing in this checkpoint -> only signals 2 and 3 will be used.")

    del model, tok
    _free_gpu()
    if report:
        report("Step 5a BERT next-sentence model")
    return scores


def compute_embedding_scores(sentence_lists, device, use_half=False, report=None):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMB_MODEL_NAME, device=device)
    if use_half and device == "cuda":
        model = model.half()

    scores = {}
    for name, sents in sentence_lists.items():
        emb = model.encode(sents, convert_to_numpy=True, normalize_embeddings=True,
                           batch_size=32, show_progress_bar=False).astype(np.float32)
        scores[name] = emb @ emb.T
        print("Embeddings done:", name)

    del model
    _free_gpu()
    if report:
        report("Step 5b sentence-embedding model")
    return scores


def compute_tfidf_scores(sentence_lists):
    scores = {}
    for name, sents in sentence_lists.items():
        plain = [re.sub(r"[।,?!.;:\"'()\-]", " ", s) for s in sents]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        X = vec.fit_transform(plain)
        scores[name] = (X @ X.T).toarray().astype(np.float32)
    print("TF-IDF done for all files.")
    return scores


def zscore(M):
    n = M.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    mean, std = M[off_diag].mean(), M[off_diag].std()
    if std < 1e-9:
        std = 1.0
    Z = (M - mean) / std
    np.fill_diagonal(Z, 0.0)
    return Z


def path_cost(path, cost):
    return sum(cost[path[k]][path[k + 1]] for k in range(len(path) - 1))


def greedy_path(start, cost):
    n = len(cost)
    path, left = [start], set(range(n))
    left.remove(start)
    while left:
        cur = path[-1]
        nxt = min(left, key=lambda j: cost[cur][j])
        path.append(nxt)
        left.remove(nxt)
    return path


def improve_path(path, cost):
    best, best_c, n = path[:], path_cost(path, cost), len(path)
    while True:
        found = False
        for seg_len in (1, 2, 3, 4):
            for i in range(n - seg_len + 1):
                block = best[i:i + seg_len]
                rest = best[:i] + best[i + seg_len:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    candidate = rest[:j] + block + rest[j:]
                    c = path_cost(candidate, cost)
                    if c < best_c - 1e-9:
                        best, best_c, found = candidate, c, True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return best


def find_best_order(score, n_starts=5):
    n = score.shape[0]
    if n <= 2:
        return list(range(n))
    cost = (-score).tolist()
    starts = []
    for s in range(n):
        p = greedy_path(s, cost)
        starts.append((path_cost(p, cost), p))
    starts.sort(key=lambda x: x[0])
    best_path, best_c = None, float("inf")
    for _, p in starts[:n_starts]:
        p = improve_path(p, cost)
        c = path_cost(p, cost)
        if c < best_c:
            best_path, best_c = p, c
    return best_path


def order_all(sentence_lists, device, w_nsp=1.0, w_emb=1.0, w_tfidf=1.0, use_half=False, report=None):
    nsp = compute_nsp_scores(sentence_lists, device, use_half, report)
    emb = compute_embedding_scores(sentence_lists, device, use_half, report)
    tfidf = compute_tfidf_scores(sentence_lists)

    ordered = {}
    for name, sents in sentence_lists.items():
        total = w_emb * zscore(emb[name]) + w_tfidf * zscore(tfidf[name])
        if name in nsp:
            total = total + w_nsp * zscore(nsp[name])
        order = find_best_order(total)
        ordered[name] = [sents[i] for i in order]
        print(f"{name}: ordered {len(order)} sentences")
    return ordered
