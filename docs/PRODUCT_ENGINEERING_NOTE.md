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

- Generating relevant suggested newsletter angles.
- Explicit separation of facts, analysis, and speculation.
- Comparison across sources instead of isolated summaries.
- A practical writing review with writer-controlled edits.
- Local Ollama inference so the demo has no paid API dependency.
- Graceful URL extraction failures and manual paste fallback.

### Deliberately left out

- Authentication, authorization and a persistent database.
- A browser extension. The current input box is the smallest reliable equivalent
  for a six-to-eight-hour prototype.

### Usefulness measures

- Time from source collection to a publishable story angle.
- Editor acceptance rate for writing suggestions.
- Number of briefs that produce a distinct angle rather than a list of summaries.
- Editor-rated usefulness of the final angle and open questions.

## Engineering

### Structure

- `app.py` owns the Streamlit shell and tabs.
- `src/research/ingest.py` parses URLs and pasted sources.
- `src/research/extractor.py` fetches URLs using HTTPX, attempts extraction with Trafilatura, falls back to BeautifulSoup, and checks for publisher RSS/Atom feeds when needed.
- `src/research/chunker.py` preserves paragraph boundaries.
- `src/research/source_summarizer.py` extracts source-linked facts and exact excerpts.
- `src/research/brief_generator.py` performs comparative synthesis.
- `src/writing/analyzer.py` runs spaCy-based deterministic checks.
- `src/writing/ollama_analyzer.py` runs contextual Qwen checks.
- `src/writing/editing.py` applies immutable-original, offset-safe edits.

### AI versus deterministic logic

Deterministic logic handles URL validation, extraction status,
chunking, sentence segmentation, dependency-based passive voice,
filler phrases, writing metrics, suggestion validation,
offset checking, and overlap detection.

Qwen2.5:3B through Ollama handles source evidence extraction,
comparative synthesis, jargon detection, paragraph clarity,
and semantic repetition analysis.

### Failure handling

HTTP failures, timeouts, blocked responses, and unsuccessful extraction receive explicit statuses. Redirects are followed. Paywalled or JavaScript-dependent pages may require manual content input. The extractor does not attempt to bypass publisher protections. Empty or navigation-only pages are not sent for meaningful source analysis. If Qwen is unavailable, deterministic writing checks remain available; research generation reports the failure.

### Evidence integrity

Each extracted evidence item stores a source ID, article title, URL, claim, category,
and exact excerpt. The brief parser accepts only excerpts already present in the
source summary. Synthesis is instructed to compare measured outcomes—not merely topics—
and to avoid treating repeated coverage as independent evidence.

### Next day of work

- Extend caching to source analysis so previously analyzed articles can be reused across brief generations, avoiding redundant model inference when the source content and model configuration have not changed.
- Improve local-model latency through batching and a smaller, faster model for first-pass extraction, while retaining the primary model for more detailed analysis.

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

I used an AI coding assistant such as ChatGpt, Copilot and Claude during implementation and personally inspected, tested, and iterated on the application code. The application uses Streamlit, spaCy, Trafilatura, HTTPX, BeautifulSoup, and local Ollama/Qwen2.5:7B.
