# The Editor

The Editor is a local Streamlit tool for turning a small collection of
readable sources into an evidence-backed editorial brief, then improving the resulting
draft without taking control away from the writer.

It is designed for a newsletter editor who needs to answer:

- What happened across the sources?
- Why does it matter in practice?
- Who should care?
- Which claims are directly supported by the sources?
- What cross-source story or newsletter angle is worth exploring?

The application uses a local Ollama model rather than a paid API. Source extraction,
evidence linking, URL validation, and writing metrics use deterministic application logic.

## What the application does

### Research Assistant

The Research Assistant accepts at least three source entries:

- HTTP or HTTPS URLs, one per line
- Pasted article, paper, announcement, or discussion text
- Multiple pasted sources separated by `---SOURCE---`

The workflow:

1. Validates and extracts each source.
2. Preserves source status, extraction method, HTTP status, failure reason, and word count.
3. Summarizes each readable source independently.
4. Extracts source-attributed claims, exact evidence excerpts, caveats, methodology,
   population, sample size, date, and implications.
5. Produces a compact cross-source synthesis.
6. Generates an optional evidence-grounded newsletter angle using stable evidence IDs.

The brief presents:

- What Happened
- Most Significant Finding
- Why It Matters
- Who Should Care
- Individual source findings
- Source comparison
- Source-attributed claims
- Analysis and practical implications
- Speculation and open questions
- Exact evidence excerpts and source links
- Newsletter angles

The synthesis model returns four compact fields:

```json
{
  "overall_finding": "What the sources collectively establish",
  "source_relationship": "How the sources agree, differ, or complement one another",
  "editorial_insight": "The most important cross-source insight",
  "practical_implications": [
    "Founders: an evidence-supported implication",
    "Builders and business operators: another evidence-supported implication"
  ]
}
```

Python maps these fields into the existing brief structure. Source evidence is never
recreated from model output: the application keeps the original validated excerpts and
their source attribution. Unknown evidence IDs are rejected.

If a URL is blocked, requires JavaScript, requires authentication, or is behind a
subscription, paste the source text manually. A failed source does not silently become
evidence.

### Writing Assistant

The Writing Assistant lets a writer paste a draft and review actionable feedback without
changing the original automatically. It combines deterministic checks with optional local
Qwen analysis for issues such as:

- Long or difficult sentences
- Passive voice
- Filler or unnecessary wording
- Jargon and paragraph clarity

It also shows word count, sentence count, average sentence length, reading time,
readability guidance, and passive-sentence count. Suggestions can be accepted or rejected,
and the original and revised drafts remain comparable.

Readability metrics are directional signals, not definitions of writing quality.

## Run locally

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The Writing Assistant uses spaCy for sentence segmentation and dependency parsing. Install
its English model once:

```bash
.venv/bin/python -m spacy download en_core_web_sm
```

### 3. Install and start Ollama

Install [Ollama](https://ollama.com), start it using the normal Ollama application or
service, and download the default local model:

```bash
ollama pull qwen2.5:3b
```

Do not start a second Ollama server if Ollama is already listening on port `11434`.

### 4. Start Streamlit

```bash
.venv/bin/streamlit run app.py
```

## Local model configuration

Create a `.env` file when you need to override the defaults:

```env
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_NUM_THREADS=4
OLLAMA_NUM_CTX=4096
RESEARCH_SOURCE_WORKERS=1
RESEARCH_CHUNK_WORKERS=1
RESEARCH_SUMMARY_TIMEOUT_SECONDS=180
RESEARCH_CHUNK_CHARACTERS=4500
RESEARCH_SYNTHESIS_TIMEOUT_SECONDS=360
```

The default configuration favors reliable local inference on a memory- and
thermally-constrained laptop. Source and chunk workers are kept serial by default.
Increasing concurrency may improve throughput on a stronger machine, but it can also
increase memory use, heat, and timeout risk.

To check the local model connection:

```bash
.venv/bin/python - <<'PY'
from src.llm.ollama import chat_json

print(
    chat_json(
        [{"role": "user", "content": "Return a JSON object containing an OK message."}],
        timeout=30,
        max_tokens=20,
        temperature=0,
    )
)
PY
```

If Ollama is unavailable, the application reports the failure. The Writing Assistant can
still provide deterministic checks, but research synthesis requires a working local model.

## Evidence and editorial safeguards

The project is intentionally conservative about unsupported claims:

- Exact excerpts remain attached to their originating source.
- Source-reported claims are distinguished from directly stated facts where the source
  data supports that distinction.
- Model-generated URLs and excerpts are not trusted as replacements for stored evidence.
- Newsletter angles must reference available stable evidence IDs.
- Missing synthesis fields are surfaced rather than filled with generic prose.
- Analysis, speculation, and open questions are presented separately from source claims.
- A source link is retained alongside the source finding whenever one exists.

The application supports editorial judgment; it does not replace source checking,
fact-checking, or final editing.

## Tests

Run the focused Research Assistant tests:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_research_pipeline.py'
```

Run the full test suite:

```bash
.venv/bin/python -m unittest discover -s tests
```

The tests mock model responses and do not require Ollama. They cover source extraction,
evidence preservation, synthesis validation and retry behavior, stable evidence matching,
optional newsletter-angle failures, UI rendering safeguards, and Writing Assistant logic.
Mocked tests verify application behavior; they do not guarantee that every live Qwen
generation will be complete or editorially strong.

## Project deliverables

- Product and engineering note:
  [docs/PRODUCT_ENGINEERING_NOTE.md](docs/PRODUCT_ENGINEERING_NOTE.md)
- Sample research brief:
  [outputs/sample_brief.md](outputs/sample_brief.md)
- Sample newsletter output:
  [outputs/sample_newsletter.md](outputs/sample_newsletter.md)
- Writing Assistant before/after:
  [outputs/draft_before.md](outputs/draft_before.md) and
  [outputs/draft_after.md](outputs/draft_after.md)

The sample files are editorial artifacts for the take-home project. 
