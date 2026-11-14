"""Paragraph-preserving source chunking."""

from typing import List


def chunk_text(text: str, max_characters: int = 6000) -> List[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current and current_length + paragraph_length + 2 > max_characters:
            chunks.append("\n\n".join(current))
            current = []
            current_length = 0

        if paragraph_length > max_characters:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_length = 0
            chunks.extend(
                paragraph[index : index + max_characters]
                for index in range(0, paragraph_length, max_characters)
            )
            continue

        current.append(paragraph)
        current_length += paragraph_length + (2 if current_length else 0)

    if current:
        chunks.append("\n\n".join(current))
    return chunks
