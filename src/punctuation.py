"""
punctuation.py
--------------
Step 1 of the project: add punctuation and split the text into sentences.

  * a pretrained multilingual punctuation model ('punctuators', pcs_47lang) finds the
    sentence ends and the commas (it runs on the CPU -> 0 GB GPU memory)
  * our own rules then choose the ending mark: ?  !  or the Hindi full stop ।
  * if the pretrained model cannot be loaded, a simple rule-based splitter is the backup
"""

PUNCT_CHARS = ",.।?!;:\"'()-"

# ---- words used by the rules for ? and ! ----
QUESTION_WORDS = {"क्या", "कौन", "कब", "क्यों", "कैसे", "कैसा", "कैसी", "कहाँ", "कहां", "किसने",
                  "किसे", "किसको", "किसका", "किसकी", "किधर", "कितना", "कितनी", "कितने"}
EXCLAIM_START = {"अरे", "ओह", "वाह", "हाय", "शाबाश", "काश", "ओए", "अबे", "उफ़", "उफ", "अहा",
                 "हे", "छि", "अरी", "ऐ"}
COMMAND_WORDS = {"जाओ", "भागो", "रुको", "चुप", "सुनो", "देखो", "आओ", "चलो", "बचाओ", "निकल",
                 "ठहरो", "बैठो", "उठो", "छोड़ो", "बस", "खबरदार"}

# ---- words used by the BACKUP splitter ----
END_WORDS = {"है", "हैं", "था", "थे", "थी", "थीं", "गया", "गए", "गई", "गयी", "होगा", "होगी", "होंगे",
             "रहा", "रहे", "रही", "सकता", "सकती", "सकते", "चाहिए", "जाता", "जाती", "जाते", "हुआ",
             "हुई", "हुए", "दिया", "किया", "लिया"}
NOT_START_WORDS = {"कि", "तो", "और", "तथा", "या", "भी", "ही", "है", "हैं", "था", "थे", "थी", "थीं",
                   "गया", "गए", "गई", "गयी", "होगा", "होगी", "होंगे", "रहा", "रहे", "रही", "हो",
                   "होता", "होती", "होते", "सकता", "सकती", "सकते", "जाता", "जाती", "जाते", "चुका",
                   "चुके", "चुकी", "करता", "करती", "करते", "गये", "जा", "दिया", "लिया", "किया",
                   "पड़ा", "पड़ी", "पड़े", "वाला", "वाली", "वाले", ""}


def fix_ending(sentence):
    """Choose the ending mark of a sentence: ?  !  or ।"""
    s = sentence.strip().rstrip(".।").rstrip()
    if not s:
        return s
    if s[-1] in "?!":                          # the model already chose ? or !, keep it
        return s
    words = [w.strip(PUNCT_CHARS) for w in s.split()]
    words = [w for w in words if w]
    if not words:
        return s + "।"
    # Question: has a question word (but not inside an indirect "... कि ..." part)
    for k, w in enumerate(words):
        if w in QUESTION_WORDS and "कि" not in words[:k]:
            return s + "?"
    # Question tag at the end: "... है ना", "... हो न"
    if len(words) >= 2 and words[-1] in {"ना", "न"} and words[-2] in {"है", "हैं", "हो", "था", "थी", "होगा"}:
        return s + "?"
    # Exclamation: starts with an interjection, or is a short command
    if words[0] in EXCLAIM_START:
        return s + "!"
    if len(words) <= 4 and any(w in COMMAND_WORDS for w in words):
        return s + "!"
    return s + "।"


def rule_based_sentences(text, min_words=5):
    """BACKUP splitter (used only if the pretrained model is unavailable)."""
    words, sentences, current = text.split(), [], []
    for k, w in enumerate(words):
        current.append(w)
        nxt = words[k + 1] if k + 1 < len(words) else ""
        if w in END_WORDS and nxt not in NOT_START_WORDS and len(current) >= min_words:
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return sentences


def load_punctuation_model():
    """Load the pretrained punctuation model. Returns None if it cannot be loaded."""
    try:
        from punctuators.models import PunctCapSegModelONNX
        model = PunctCapSegModelONNX.from_pretrained("pcs_47lang")
        print("Pretrained punctuation model loaded.")
        return model
    except Exception as e:
        print("WARNING: could not load the pretrained punctuation model ->", e)
        print("Using the simple backup splitter instead (results will be less accurate).")
        return None


def add_punctuation(text, punct_model=None):
    """Raw Hindi text (no punctuation) -> list of punctuated sentences."""
    sentences = None
    if punct_model is not None:
        try:
            sentences = punct_model.infer([text], apply_sbd=True)[0]
        except Exception as e:
            print("WARNING: punctuation model failed on one file ->", e, "| using backup splitter for it.")
    if sentences is None:
        sentences = rule_based_sentences(text)
    return [fix_ending(s) for s in sentences if s.strip()]


def punctuate_all(clean_texts):
    """Run punctuation on every file. Returns {'sample-1': [sentence, ...], ...}"""
    punct_model = load_punctuation_model()
    result = {}
    for name, text in clean_texts.items():
        result[name] = add_punctuation(text, punct_model)
        print(f"{name}: {len(result[name])} sentences")
    return result
