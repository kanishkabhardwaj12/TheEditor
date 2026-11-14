# Product and Engineering Note

## Product thinking

### Primary user

The primary user is a newsletter editor who regularly turns a messy set of articles,
announcements, and research into a short, source-backed story for founders, builders,
and business operators.

### Problem selected

The highest-value problem is the transition from “I collected several links” to “I
understand what is actually supported, what is interpretation, and what is worth
writing about.” A generic summarizer saves reading time but does not reliably surface
editorial tension or protect against overclaiming.

### Core workflow

1. Paste at least three URLs or source texts.
2. Extract article content, retaining source title, URL, status, and extraction method.
3. Use Qwen locally to extract source-grounded facts, caveats, claims to verify, and
   exact supporting excerpts.
4. Synthesize a comparative brief with facts, analysis, uncertainty, implications,
   open questions, evidence links, and distinct newsletter angles.
5. Paste the resulting draft into the Writing Assistant.
6. Review deterministic and Qwen feedback, then accept or reject edits individually.

### Prioritized features

- Source-grounded evidence excerpts and clickable links.
- Explicit separation of facts, analysis, and speculation.
- Comparison across sources instead of isolated summaries.
- A practical writing review with writer-controlled edits.
- Local Ollama inference so the demo has no paid API dependency.
- Graceful URL extraction failures and manual paste fallback.

### Deliberately left out

- Authentication, collaboration, publishing, and a persistent database.
- Browser automation for paywalls or JavaScript-heavy sites.
- Automatic publication or blind whole-document rewriting.
- A browser extension. The current input box is the smallest reliable equivalent
  for a six-to-eight-hour prototype; a Pocket-style connector would be the next
  integration.

### Usefulness measures

- Time from source collection to a publishable story angle.
- Percentage of factual claims with a traceable source excerpt.
- Editor acceptance rate for writing suggestions.
- Rejection rate for suggestions judged stylistic or meaning-changing.
- Number of briefs that produce a distinct angle rather than a list of summaries.
- Editor-rated usefulness of the final angle and open questions.

## Engineering

### Structure

- `app.py` owns the Streamlit shell and tabs.
- `src/research/ingest.py` parses URLs and pasted sources.
- `src/research/extractor.py` uses Trafilatura first, then HTTPX/BeautifulSoup and
  official feed fallback.
- `src/research/chunker.py` preserves paragraph boundaries.
- `src/research/source_summarizer.py` extracts source-linked facts and exact excerpts.
- `src/research/brief_generator.py` performs comparative synthesis.
- `src/writing/analyzer.py` runs spaCy-based deterministic checks.
- `src/writing/ollama_analyzer.py` runs contextual Qwen checks.
- `src/writing/editing.py` applies immutable-original, offset-safe edits.

### AI versus deterministic logic

Deterministic logic handles URL validation, extraction status, chunking, character
offsets, overlap detection, sentence segmentation, dependency-based passive voice,
filler phrases, metrics, and rewrite validation.

Qwen2.5:7B through Ollama handles source evidence extraction, comparative synthesis,
jargon, paragraph clarity, semantic repetition, and awkward phrasing. The application
does not use OpenAI, Gemini, Groq, or paid APIs.

### Failure handling

Blocked, timed-out, redirected, paywalled, and JavaScript-dependent URLs receive an
explicit status. The extractor does not repeatedly retry blocked pages or attempt to
bypass protections. Editors can paste inaccessible content manually. Empty or
navigation-only pages never reach Qwen. If Qwen is unavailable, deterministic writing
checks remain available; research generation reports the failure.

### Evidence integrity

Each extracted evidence item stores a source ID, article title, URL, claim, category,
and exact excerpt. The brief parser accepts only excerpts already present in the
source summary. Synthesis is instructed to compare measured outcomes—not merely topics—
and to avoid treating repeated coverage as independent evidence.

### Next day of work

- Add a Pocket/RSS ingestion connector and persistent brief history.
- Add a human-facing claim matrix showing which source supports each statement.
- Add browser rendering only for sources where the publisher explicitly permits it.
- Add evaluation fixtures with known source conflicts and editor-labeled rewrites.
- Improve local-model latency with batching and a smaller fast model for first-pass
  extraction.

## Editorial judgment

Clear writing gives readers the main point early, supplies enough context to interpret
it, uses concrete language, and makes uncertainty visible. It does not remove every
technical term or force every sentence into the same rhythm.

The Writing Assistant should be evaluated by whether editors accept useful changes,
reject meaning-changing changes, and report that the revised draft is clearer without
losing nuance. Suggestions should be compared against the immutable original and
reviewed with expert judgment, not a readability score alone.

The tool should avoid automatically changing factual claims, proper nouns, technical
terms that are clear in context, intentional passive voice, quotations, tone, and
sentences whose length is justified by their structure.

## Tools and disclosure

I used an AI coding assistant during implementation and personally inspected, tested,
and iterated on the application code. The application uses Streamlit, spaCy,
Trafilatura, HTTPX, BeautifulSoup, and local Ollama/Qwen2.5:7B.
