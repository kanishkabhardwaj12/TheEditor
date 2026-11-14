"""Deterministic writing checks backed by spaCy."""

import re
from typing import Dict, List, Optional

import spacy
from spacy.language import Language

from src.models.schemas import WritingSuggestion

FILLER_PHRASES: Dict[str, str] = {
    "in order to": "Use 'to' instead.",
    "basically": "Remove it unless it adds a specific qualification.",
    "actually": "Remove it unless the contrast is important.",
    "very": "Use a more precise adjective or remove it.",
    "really": "Use a more precise description or remove it.",
    "just": "Remove it unless it changes the meaning.",
}

_NLP: Optional[Language] = None


def _get_nlp() -> Language:
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model en_core_web_sm is not installed. Run: "
                "python -m spacy download en_core_web_sm"
            ) from exc
    return _NLP


def _sentence_rewrite(sentence: str, doc_sentence) -> Optional[str]:
    conjunction = re.search(r",\s+(and|but|so)\s+", sentence, re.I)
    if conjunction:
        first = sentence[: conjunction.start()].rstrip(", ")
        second = sentence[conjunction.end() :].strip()
        second = re.sub(r"^(and|but|so)\s+", "", second, flags=re.I)
        if (
            first
            and second
            and _has_finite_verb(first)
            and _has_finite_verb(second)
            and _has_independent_clause(first)
            and _has_independent_clause(second)
        ):
            return f"{first}. {second[0].upper()}{second[1:]}"
    return None


def _has_finite_verb(text: str) -> bool:
    return any(token.tag_ in {"VB", "VBD", "VBP", "VBZ", "MD"} for token in _get_nlp()(text))


def _has_independent_clause(text: str) -> bool:
    doc = _get_nlp()(text)
    root = doc[:].root
    return root is not None and any(
        child.dep_ in {"nsubj", "nsubjpass", "expl"} for child in root.children
    )


def _suggestion_is_valid(text: str, suggestion: str, start: int, end: int) -> bool:
    if not suggestion.strip() or suggestion.strip() == text[start:end].strip():
        return False
    if not text[start:end].strip():
        return False
    candidate = _get_nlp()(suggestion.strip())
    if not list(candidate.sents) or not any(
        token.pos_ in {"VERB", "AUX"} for token in candidate
    ):
        return False
    if suggestion.count("(") != suggestion.count(")") or suggestion.count('"') % 2:
        return False
    return True


def _passive_rewrites(sentence: str, doc_sentence):
    results = []
    for passive_subject in (
        token for token in doc_sentence if token.dep_ == "nsubjpass"
    ):
        root = passive_subject.head
        agent = next(
            (
                token
                for token in doc_sentence
                if token.dep_ == "agent" and token.head == root
            ),
            None,
        )
        if agent is None:
            continue
        span_start = passive_subject.left_edge.idx - doc_sentence.start_char
        span_end = agent.right_edge.idx + len(agent.right_edge.text) - doc_sentence.start_char
        passive_clause = sentence[span_start:span_end]
        subject = sentence[
            agent.left_edge.idx - doc_sentence.start_char : agent.right_edge.idx
            + len(agent.right_edge.text)
            - doc_sentence.start_char
        ]
        subject = re.sub(r"^by\s+", "", subject, flags=re.I).strip()
        if not subject:
            continue
        object_text = list(passive_subject.subtree)
        if not object_text:
            continue
        object_start = min(token.idx for token in object_text) - doc_sentence.start_char
        object_end = max(token.idx + len(token.text) for token in object_text) - doc_sentence.start_char
        obj = sentence[object_start:object_end].strip()
        irregular = {"write": "wrote", "lead": "led", "make": "made", "build": "built"}
        verb = root.lemma_
        if root.tag_ == "VBD" or root.morph.get("Tense") == ["Past"]:
            verb = irregular.get(verb, f"{verb}ed")
        obj = obj[:1].lower() + obj[1:] if obj else obj
        rewrite = f"{subject[:1].upper() + subject[1:]} {verb} {obj}."
        results.append((span_start, span_end, passive_clause, rewrite))
    return results


def _phrase_rewrite(sentence: str, phrases: List[str]) -> str:
    rewritten = sentence
    for phrase in phrases:
        replacement = "to" if phrase == "in order to" else ""
        rewritten = re.sub(rf"\b{re.escape(phrase)}\b", replacement, rewritten, flags=re.I)
    rewritten = re.sub(r"\s{2,}", " ", rewritten)
    rewritten = re.sub(r"\s+([,.!?])", r"\1", rewritten).strip()
    return rewritten[:1].upper() + rewritten[1:] if rewritten else rewritten


def analyze_draft(text: str) -> List[WritingSuggestion]:
    doc = _get_nlp()(text)
    suggestions: List[WritingSuggestion] = []
    for sentence in doc.sents:
        sentence_text = sentence.text
        start = sentence.start_char
        end = sentence.end_char
        non_space_tokens = [token for token in sentence if not token.is_space]
        subordinate_count = sum(
            token.dep_ in {"advcl", "ccomp", "xcomp", "relcl"} for token in non_space_tokens
        )
        clause_count = sum(
            token.dep_ in {"conj", "advcl", "ccomp", "xcomp", "relcl"}
            for token in non_space_tokens
        )
        if len(non_space_tokens) > 28 and (subordinate_count >= 2 or clause_count >= 3):
            rewrite = _sentence_rewrite(sentence_text, sentence)
            if rewrite and _suggestion_is_valid(text, rewrite, start, end):
                suggestions.append(
                    WritingSuggestion(
                        "Long sentence",
                        sentence_text,
                        f"This sentence combines {clause_count + 1} clause-level ideas, "
                        "which makes the main point harder to follow.",
                        rewrite,
                        start,
                        end,
                        "medium",
                        0.88,
                        suggestion_id=f"{start}:{end}:long",
                    )
                )

        passive_detected = any(
            token.dep_ in {"nsubjpass", "auxpass"} for token in sentence
        )
        if passive_detected:
            for relative_start, relative_end, passive_clause, rewrite in _passive_rewrites(
                sentence_text, sentence
            ):
                passive_start = start + relative_start
                passive_end = start + relative_end
                if not _suggestion_is_valid(text, rewrite, passive_start, passive_end):
                    continue
                suggestions.append(
                    WritingSuggestion(
                        "Passive voice",
                        passive_clause,
                        "This sentence uses a passive construction with an explicit "
                        "agent ('by ...'). Consider the active version if the actor "
                        "should be more prominent; passive voice is not inherently wrong.",
                        rewrite,
                        passive_start,
                        passive_end,
                        "medium",
                        0.84,
                        suggestion_id=f"{passive_start}:{passive_end}:passive",
                    )
                )

        lowered = sentence_text.lower()
        found = [phrase for phrase in FILLER_PHRASES if re.search(rf"\b{re.escape(phrase)}\b", lowered)]
        if found:
            for phrase in found:
                phrase_match = re.search(rf"\b{re.escape(phrase)}\b", sentence_text, re.I)
                if not phrase_match:
                    continue
                phrase_start = start + phrase_match.start()
                phrase_end = start + phrase_match.end()
                replacement = "to" if phrase == "in order to" else ""
                suggestions.append(
                    WritingSuggestion(
                    "Filler phrase",
                    phrase,
                    FILLER_PHRASES[phrase],
                    replacement,
                    phrase_start,
                    phrase_end,
                    "low",
                    0.78,
                    suggestion_id=f"{phrase_start}:{phrase_end}:filler",
                )
                )
    return suggestions
