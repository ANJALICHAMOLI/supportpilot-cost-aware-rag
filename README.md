# SupportPilot — Cost-Aware RAG Support Copilot

A small, fully-working AI application that answers customer-support questions by retrieving relevant passages from uploaded documents (RAG), routes each question to a cheaper or stronger LLM based on complexity, tracks simulated cost and real latency for every request, enforces a spending budget, and evaluates itself against a fixed test set.

**Runs entirely on free tiers — $0 to build, run, or demo.**

Groq and Google Gemini provide free API access, while embeddings run locally on the CPU using an open-source model. No paid infrastructure is required for this project.

Built as an interview-ready portfolio project — every design decision is intended to be understandable and defensible rather than simply using buzzwords.

---

## 1. Problem

Naively wiring an LLM directly to a chat UI creates two problems for a customer-support system:

1. The model can confidently hallucinate when the required information is not available.
2. Using the strongest model for every question increases inference cost even when the question is only a simple factual lookup.

---

## 2. Solution

- **RAG (Retrieval-Augmented Generation):** Answers are grounded in uploaded documents, with retrieved sources shown alongside every answer. The model is instructed to avoid inventing information that is not present in the documents.

- **Cost-aware model routing:** A lightweight rule-based classifier sends simple, single-fact questions to a fast Groq model (`llama-3.1-8b-instant`) and reserves Google Gemini (`gemini-flash-latest`) for questions that require more reasoning or synthesis.

- **Budget enforcement:** Before an expensive model call, the application checks whether the estimated cost would exceed the configured session budget. If necessary, it gracefully downgrades to the cheaper model instead of blocking the request.

- **Observability:** Every request records tokens, estimated cost, latency, and success/failure in a CSV log, which is visualized in a dashboard.

- **Evaluation harness:** A fixed set of 25 test questions, including hallucination traps and unanswerable questions, is run through the actual pipeline to produce repeatable quality metrics.

---

## 3. Architecture

```text
                         User
                           |
                           v
                    Streamlit UI
                       (app.py)
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
    Document Upload   Question Asked   Evaluation
          |                |            Harness
          v                v                |
    Load → Split       Retriever <-------+
    → Embed → Store    (vector search,
       (Chroma)           top-k)
          |                |
          +-------+--------+
                  |
                  v
           Query Router
          (rule-based:
        cheap vs strong)
                  |
                  v
            Budget Check
        (downgrade if needed)
                  |
                  v
          Selected LLM
          + Context
                  |
                  v
          Answer + Sources
                  |
                  v
             CSV Logger
       (tokens, cost, latency,
          success/failure)
                  |
                  v
       Cost / Latency Dashboard