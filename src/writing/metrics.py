"""Writing metrics calculations."""

import math
import re
from typing import Dict, Union

import spacy

_NLP = None

def _sentence_doc(text: str):
    global _NLP
    if _NLP is not None:
        return _NLP(text)
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        _NLP = spacy.blank("en")
        _NLP.add_pipe("sentencizer")
    return _NLP(text)


def calculate_metrics(text: str) -> Dict[str, Union[float, int, str]]:
    words = re.findall(r"\b\w+\b", text)
    doc = _sentence_doc(text)
    sentences = list(doc.sents)

    word_count = len(words)
    sentence_count = max(1, len(sentences))
    avg_sentence_length = word_count / sentence_count
    reading_time_minutes = word_count / 200
    reading_time_display = max(1, math.ceil(reading_time_minutes))

    readability_hint = "Easy to scan"
    if avg_sentence_length > 22:
        readability_hint = "Dense/technical"
    elif avg_sentence_length > 16:
        readability_hint = "Moderate"

    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "passive_sentence_count": sum(
            any(token.dep_ in {"nsubjpass", "auxpass"} for token in sentence)
            for sentence in sentences
        ),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "reading_time_minutes": reading_time_display,
        "readability_hint": readability_hint,
    }
