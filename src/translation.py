import gc
import re
import time

import torch

MT_MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"


GLOSSARY = {

    "गढ़ मुक्तेश्वर": "Garhmukteshwar", "मुक्तेश्वर": "Mukteshwar", "बल्लभपुर": "Ballabhpur",
    "बनारस छावनी": "Banaras Cantonment", "बनारस": "Banaras", "वाराणसी": "Varanasi", "काशी": "Kashi",
    "इलाहाबाद": "Allahabad", "प्रयाग": "Prayag", "अयोध्या": "Ayodhya", "मथुरा": "Mathura",
    "उज्जैन": "Ujjain", "द्वारका": "Dwarka", "हरिद्वार": "Haridwar", "कांचीपुरम": "Kanchipuram",
    "कांजीवरम": "Kanjeevaram", "चेंगलपट": "Chengalpattu", "अरकोनम": "Arakkonam", "चेन्नै": "Chennai",
    "भटनी": "Bhatni", "छपरा": "Chhapra", "मुगलसराय": "Mughalsarai", "लखनऊ": "Lucknow",
    "गाजियाबाद": "Ghaziabad", "मेरठ": "Meerut", "दिल्ली": "Delhi", "हस्तिनापुर": "Hastinapur",
    "तिजारा": "Tijara", "शिमला": "Shimla", "भोपाल": "Bhopal", "मुंबई": "Mumbai", "कोटा": "Kota",
    "आगरा": "Agra", "गोरखपुर": "Gorakhpur", "अमृतसर": "Amritsar", "देहरादून": "Dehradun",
    "लक्सर": "Laksar", "महू": "Mhow", "धनुष्कोटि": "Dhanushkodi", "पुष्कर": "Pushkar",
    "बद्रीनाथ": "Badrinath", "अलवर": "Alwar", "कश्मीरी गेट": "Kashmiri Gate", "खांडवी": "Khandavi",
    "नागपुर": "Nagpur", "मणिपुर": "Manipur", "गजपुर": "Gajapur", "पोदनपुर": "Podanpur",
    "असम": "Assam", "चीन": "China", "जापान": "Japan", "कर्नाटक": "Karnataka", "तमिलनाडु": "Tamil Nadu",
    "राजस्थान": "Rajasthan", "पंजाब": "Punjab", "हिमाचल": "Himachal", "बिहार": "Bihar",

    "गंगा": "Ganga", "यमुना": "Yamuna", "सरयू": "Sarayu",

    "परशुराम": "Parashurama", "श्रीकृष्ण": "Shri Krishna", "कृष्ण": "Krishna", "राम": "Ram",
    "शिवि": "Shivi", "युधिष्ठिर": "Yudhishthira", "अर्जुन": "Arjuna", "महाकाल": "Mahakal",
    "महाभारत": "Mahabharata", "महावीर": "Mahavir", "आदिनाथ": "Adinath", "शांतिनाथ": "Shantinath",
    "कुंथनाथ": "Kunthunath", "अरहनाथ": "Arahnath", "मल्लिनाथ": "Mallinath", "पार्श्वनाथ": "Parshvanath",
    "श्रेयंषकुमार": "Shreyanshkumar", "सरस्वती": "Saraswati", "बिहारी लाल": "Bihari Lal",
    "कैलाशनाथ": "Kailashnath", "चौधरी": "Chaudhary", "राहुल": "Rahul", "अंजलि": "Anjali",
    "अमन": "Aman", "टीना": "Tina", "रिफ़त": "Rifat", "राजेश": "Rajesh",
}


def protect_names(sentence):
    for hindi in sorted(GLOSSARY, key=len, reverse=True):
        sentence = re.sub(r"(?<!\S)" + re.escape(hindi) + r"(?=[\s।,?!;:]|$)", GLOSSARY[hindi], sentence)
    return sentence


def make_chunks(sentences, size=5):
    return [sentences[i:i + size] for i in range(0, len(sentences), size)]


def load_translator(device, use_half=False):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MT_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL_NAME).to(device).eval()
    if use_half and device == "cuda":
        model = model.half()
    return tok, model


@torch.no_grad()
def translate_sentences(sentences, tok, model, device, batch_size=8):
    english = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
        out = model.generate(**enc, num_beams=4, max_new_tokens=150, no_repeat_ngram_size=3)
        english.extend(tok.batch_decode(out, skip_special_tokens=True))
    return english


def translate_all(ordered_sentences, device, chunk_size=5, use_half=False, report=None):
    tok, model = load_translator(device, use_half)
    start = time.time()
    output = {}

    for name, sentences in ordered_sentences.items():
        hindi_chunks = make_chunks(sentences, chunk_size)
        english_chunks = []
        for k, chunk in enumerate(hindi_chunks, start=1):
            english = translate_sentences([protect_names(s) for s in chunk], tok, model, device)
            english_chunks.append(" ".join(e.strip() for e in english))
            print(f"  {name}: chunk {k}/{len(hindi_chunks)} done ({time.time() - start:.0f}s elapsed)")
        output[name] = {"chunks_hi": [" ".join(c) for c in hindi_chunks],
                        "chunks_en": english_chunks}
        print(f"{name}: {len(hindi_chunks)} chunks translated")

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if report:
        report("Step 6 translation model")
    return output
