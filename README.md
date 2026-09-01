# SupportPilot AI — Cost-Aware RAG Support Copilot

A small, fully-working AI application that answers customer-support questions by
retrieving relevant passages from uploaded documents (RAG), routes each question to
a cheaper or stronger LLM based on complexity, tracks (simulated) cost and real
latency for every request, enforces a spending budget, and evaluates itself against
a fixed test set.

**Runs entirely on free tiers — $0 to build, run, or demo.** Groq and Google
Gemini both offer genuinely free API keys (no credit card), and embeddings run
locally on your own CPU with an open-source model, so there's no billing
relationship with anyone required to use this project at all.

Built as an interview-ready portfolio project — every design decision below is one
I can explain and defend, not just a checklist of buzzwords.

---

## 1. Problem

Naively wiring an LLM directly to a chat UI has two problems for a real support
tool: (1) it will confidently make things up (hallucinate) when it doesn't actually
know something, and (2) if you always call your most powerful (most expensive)
model for every question — including trivial ones — costs scale linearly with
traffic for no quality benefit on the easy questions.

## 2. Solution

- **RAG (Retrieval-Augmented Generation)**: answers are grounded in the user's own
  uploaded documents, with retrieved sources shown alongside every answer, and the
  model is explicitly instructed to say "I don't know" rather than invent facts.
- **Cost-aware model routing**: a lightweight rule-based classifier sends simple,
  single-fact questions to a fast free model on Groq (`llama-3.1-8b-instant`) and
  reserves a stronger free model on Google Gemini (`gemini-flash-latest`) for
  questions that need multi-step reasoning or synthesis.
- **Budget enforcement**: before any expensive call, the app checks whether it
  would exceed a configurable session budget, and gracefully downgrades to the
  cheap model instead of blocking the user.
- **Observability**: every request logs tokens, cost, latency, and success/failure
  to a CSV, visualized in a live dashboard.
- **Evaluation harness**: a fixed set of 25 test questions (including
  hallucination traps and unanswerable questions) is run through the real
  pipeline to produce repeatable quality metrics — not just "it looked fine when I
  tried it."

## 3. Architecture

```
                 User
                   |
             Streamlit UI (app.py)
                   |
       ┌───────────┼────────────┐
       |           |            |
  Document      Question     Evaluation
  Upload        Asked        Harness
       |           |            |
   Load→Split   Retriever   (reuses the
   →Embed       (vector       same
   →Store       search,       pipeline
   (Chroma)     top-k)        below)
       |           |
       └─────┬─────┘
             |
      Query Router (rule-based:
      cheap vs strong model)
             |
      Budget Check (downgrade
      if over session limit)
             |
      Selected LLM + Context
      → Answer + Sources
             |
      Logger (CSV: tokens,
      cost, latency, success)
             |
      Cost / Latency Dashboard
```

### Why this shape, and not something more complex?
This is a **linear pipeline**, not an agent framework, not a multi-service
architecture, not a database-backed system. That's a deliberate choice: every
piece is independently understandable and testable, matching the actual scope of
the problem (single-user, small document corpus, demo-scale traffic). See
"Future Improvements" for what would change at real production scale.

## 4. Tech Stack

| Piece | Choice | Why |
|---|---|---|
| UI | Streamlit | No separate frontend needed; whole app stays in Python |
| Orchestration | LangChain | Standard loaders/splitters/prompt templates; avoids reinventing well-trodden plumbing |
| Embeddings | Local `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) | Runs on CPU, no API key, no rate limits, $0 forever |
| Vector store | Chroma | Local, persistent, zero external infra, realistic ANN-index pattern |
| Cheap LLM | Groq `llama-3.1-8b-instant` | Free tier, no credit card, extremely fast inference |
| Strong LLM | Google Gemini `gemini-flash-latest` | Free tier, stronger reasoning than the 8B model |
| Logging | CSV + pandas | Human-readable, zero setup, sufficient for single-user demo scale |
| Evaluation | Custom keyword/source-matching harness | Deterministic, free, fast — no LLM-as-judge complexity needed for fixed-fact support docs |

**Total cost to build, run, and demo this project: $0.** Both LLM providers offer
free API keys with no credit card, and the embedding model runs locally. Free
tiers ARE rate-limited (requests/minute, tokens/day per provider) — fine for
development and live demos, not intended as a production backend at scale.

## 5. RAG Pipeline (Features 1 & 2)

`Document → Load (rag/loader.py) → Split into ~500-char overlapping chunks
(rag/splitter.py) → Embed + store in Chroma (rag/embeddings.py) → Similarity
search top-k (rag/retriever.py) → Prompt with retrieved context
(rag/generator.py) → LLM → Answer + Sources`

The prompt explicitly instructs the model to answer **only** from the supplied
context and to say so plainly when the answer isn't present — directly targeting
hallucination. Retrieved chunks are always shown in the UI (never hidden) so
retrieval quality is inspectable, not a black box.

## 6. Model Routing Strategy (Feature 3)

`routing/model_router.py` uses three rule-based signals to decide cheap vs. strong:
1. Presence of complex-reasoning keywords (compare, why, explain, summarize, ...)
2. Question length (long questions tend to be multi-part)
3. Number of distinct topics mentioned (spanning multiple documents likely needs
   synthesis, not lookup)

Why rule-based instead of an ML classifier or an LLM-based router? No labeled
training data exists at this scale, and using an LLM call just to decide *which*
LLM to use would add cost and latency to every request — defeating the purpose.
The routing decision (model + human-readable reason) is always shown in the UI.

## 7. Cost & Latency Tracking (Feature 4)

Every request logs: timestamp, question, model, input/output tokens (from the
**real** provider API response, not estimated), cost, latency, and success/failure
to `request_log.csv`. The dashboard aggregates: total queries, total/average cost,
average latency, cheap vs. strong model usage %, and cost/latency broken down by
model.

**Important honesty note**: since both Groq and Gemini are used on their free
tiers, the *actual* dollar amount billed is always $0. The "cost" figures shown
are a **simulation**, computed by applying each provider's published paid-tier
per-token rate to the real token counts each free-tier call actually used. This
keeps the cost-awareness architecture meaningful to demonstrate — the same
routing, budget-check, and logging code would produce real billing numbers with
zero changes on day one of switching to a paid plan.

## 8. Budget Control (Feature 5)

`monitoring/cost_tracker.py`'s `BudgetTracker` estimates the cost of a planned
call *before* making it (using a rough token estimate, since real output length
isn't known until generation finishes). If the strong model would exceed the
remaining session budget, the request is **transparently downgraded** to the
cheap model rather than blocked outright — a graceful-degradation pattern, with
the downgrade always disclosed in the UI.

## 9. Evaluation Methodology (Feature 6)

`evaluation/test_cases.json` contains 25 hand-written questions across 6
categories: normal lookups, multi-context questions, ambiguous questions,
questions with no answer in the documents, deliberate hallucination traps, and
edge cases (empty/gibberish input). `evaluation/evaluate.py` runs every case
through the **actual production pipeline** (not a separate mock) and scores:
- **Pipeline success rate** — did the request complete without crashing?
- **Keyword answer accuracy** — does the answer contain an expected fact/phrase?
- **Source retrieval accuracy** — did retrieval pull the expected document?

Keyword/source matching was chosen over LLM-as-judge grading to keep evaluation
free, deterministic, and simple to reason about, given this project's fixed-fact
domain — an explicit and honest scope tradeoff.

## 10. Results

*(Run the Evaluation tab after uploading the sample docs in `data/sample_docs/`
and paste your actual numbers here before sharing this repo — e.g.:)*

```
Pipeline success rate: 100%
Keyword answer accuracy: XX% (over N applicable cases)
Source retrieval accuracy: XX% (over N applicable cases)
```

## 11. Limitations

- Single active knowledge base at a time (uploading new docs replaces the old
  vector store rather than merging or supporting multiple collections).
- Chunking uses a fixed character-based size, not semantic or token-based splitting.
- Routing is rule-based on surface text features, not learned from labeled data.
- CSV logging doesn't handle concurrent multi-user writes safely.
- Pricing constants in `cost_tracker.py` are hardcoded and can go stale.
- Cost figures are **simulated** (based on free-tier token usage × published paid
  pricing), not real billing — see Section 7 for why this is an honest tradeoff.
- Free-tier rate limits (requests/minute, tokens/day) mean this isn't suitable for
  high-traffic production use without upgrading to paid provider plans.
- Evaluation uses keyword matching, not nuanced semantic answer grading.
- Local embedding model (all-MiniLM-L6-v2) is lower-capacity than large hosted
  embedding APIs — fine for this narrow support-doc corpus, may need upgrading
  for more nuanced or larger-scale document collections.

## 12. Future Improvements

- Reranking retrieved chunks before generation
- Hybrid search (keyword + semantic)
- Conversation memory across turns
- Agent/tool-calling for actions beyond Q&A
- A learned (ML-based) query router instead of rule-based
- LLM-as-judge evaluation for nuanced answer quality
- Response caching for repeated questions
- Provider fallback (e.g. if Groq or Gemini is down/rate-limited, fall back to
  another free provider like Cerebras or Together AI)
- Real billing integration once/if moving off free tiers to paid usage
- A real database instead of CSV logging for multi-user production use
- Cloud deployment (this app currently targets local/single-instance use)

## 13. How to Run Locally (macOS)

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd supportpilot-ai

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Get your two FREE API keys (no credit card needed for either):
#    - Groq:   https://console.groq.com/keys
#    - Gemini: https://aistudio.google.com/apikey

# 5. Set up your keys
cp .env.example .env
# then open .env and paste both real keys

# 6. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. First run will download the local
embedding model (~80MB, one-time, then fully cached offline). Upload the sample
docs in `data/sample_docs/` (or your own PDF/TXT/MD files) in the **Document
Upload** tab, then ask questions in the **Support Copilot** tab.

**Free tier rate limits to be aware of during a live demo**: both Groq and Gemini
cap requests per minute on the free tier. If you hit a rate limit mid-demo, the
app automatically retries a couple of times with a short delay (see
`rag/generator.py`) — if it still fails, just wait ~30–60 seconds and try again.

## 14. Live Demo

*(Add your deployed Streamlit Community Cloud link here once deployed.)*

## 15. One-Paragraph Interview Summary

"I built a cost-aware RAG support copilot in Streamlit. Documents are chunked,
embedded, and stored in a local Chroma vector store; questions are answered using
only retrieved context, with sources always shown. A rule-based router sends
simple questions to a cheap model and complex ones to a stronger model, with a
budget guardrail that downgrades to the cheap model if a session spending limit
would be exceeded. Every request is logged with real token counts, cost, and
latency, visualized in a dashboard. I built a 25-case evaluation harness —
including deliberate hallucination traps and unanswerable questions — that runs
through the actual pipeline to give repeatable quality metrics instead of ad-hoc
manual testing."
