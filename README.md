# SupportPilot — Cost-Aware RAG Support Copilot

A cost-aware customer support AI application built with RAG, intelligent LLM routing, budget control, monitoring, and automated evaluation.

## 1. ✨ Features

- **RAG-based Support:** Retrieves relevant information from uploaded documents before generating answers.
- **LLM-Based Routing:** Uses Qwen on Groq to classify queries as `SIMPLE` or `COMPLEX`, routing them to an appropriate model.
- **Cost-Aware Inference:** Tracks estimated inference costs and enforces a configurable session budget.
- **Budget Fallback:** Automatically downgrades to the cheaper model when the stronger model would exceed the remaining budget.
- **Monitoring:** Logs model selection, routing reason, token usage, estimated cost, latency, and request status.
-  **Evaluation:** Includes 25 test cases covering factual queries, multi-context questions, ambiguous inputs, hallucination traps, and edge cases.
- **Grounded Generation:** Answers are restricted to retrieved document context to reduce hallucination.

## 2. 🏗️ Architecture

```text
                         User
                           |
                           v
                    Streamlit UI
                           |
              +------------+------------+
              |            |            |
              v            v            v
        Document       Question      Evaluation
         Upload          Asked         Suite
              |            |
              v            v
        Load → Split    Retriever
        → Embed         (Chroma)
        → Store            |
              |             |
              +------+------+
                     |
                     v
              Qwen LLM Router
                (Groq)
                     |
              +------+------+
              |             |
           SIMPLE        COMPLEX
              |             |
              v             v
        Qwen / Groq     Gemini Flash
              |             |
              +------+------+
                     |
                     v
              Budget Check
                     |
                     v
             Answer + Sources
                     |
                     v
               CSV Logging
                     |
                     v
            Monitoring Dashboard
```


## 4. Tech Stack

| Technology      | Purpose                              |
| --------------- | ------------------------------------ |
| 🐍 Python       | Core application                     |
| ⚡ Streamlit     | Web interface                        |
| 🦜 LangChain    | LLM/RAG orchestration                |
| 🤗 HuggingFace  | Local embeddings                     |
| 🗄️ Chroma      | Vector database                      |
| ⚡ Groq          | Fast LLM inference and query routing |
| ✨ Google Gemini | Stronger model for complex queries   |
| 🐼 Pandas       | Logging and evaluation analysis      |

<p align="left"> <img src="https://cdn.simpleicons.org/python" width="40" alt="Python"/> <img src="https://cdn.simpleicons.org/streamlit" width="40" alt="Streamlit"/> <img src="https://cdn.simpleicons.org/langchain" width="40" alt="LangChain"/> <img src="https://cdn.simpleicons.org/huggingface" width="40" alt="HuggingFace"/> <img src="https://cdn.simpleicons.org/google" width="40" alt="Google"/> <img src="https://cdn.simpleicons.org/pandas" width="40" alt="Pandas"/> </p>


## RAG Pipeline 

`Document → Document Loader → Text Splitter i → Embed →Chroma Vector Store → Similarity Retrieval → Context + Question→ Select  LLM → Answer + Sources`

The prompt explicitly instructs the model to answer **only** from the supplied
context and to say so plainly when the answer isn't present directly targeting
hallucination. Retrieved chunks are always shown in the UI (never hidden) so
retrieval quality is inspectable, not a black box.
The system uses sentence-transformers/all-MiniLM-L6-v2 to generate embeddings locally, avoiding an external embedding API.

## 🟡 Intelligent Model Routing

A lightweight Qwen model running through Groq acts as the query router.

### SIMPLE

Direct single fact questions that can usually be answered from one piece of information.

Examples:

> How many days do I have to return an item?
>
> What is your email address?
>
> What are your store hours?

### COMPLEX

Questions requiring comparison, multiple facts, reasoning, synthesis, or recommendations.

Examples:

> Compare standard and express shipping.
>
> If my return is approved, when will I get my refund and is shipping refundable?
>
> Which shipping option is better for me?

The router then selects the appropriate generation model.

## ◈ Cost & Budget Control

The application records:

- Input tokens
- Output tokens
- Estimated cost
- Latency
- Selected model
- Routing decision
- Request success
- Budget-based downgrades

Before using the stronger model, the application checks the remaining session budget. If the estimated request would exceed the budget, it falls back to the cheaper model.

Cost values are estimated using real token usage and configured model pricing.

## ⌁ Evaluation

The project includes 25 evaluation cases covering:

- Normal factual queries
- Multi-context questions
- Ambiguous questions
- Questions outside the knowledge base
- Hallucination traps
- Edge cases and gibberish input

The evaluation pipeline measures:

- **Pipeline Success Rate**
- **Keyword Answer Accuracy**
- **Source Retrieval Accuracy**

The evaluation runs through the actual RAG pipeline rather than a separate mock system.

## ⚒ Fixes & Improvements

During evaluation and testing, several issues were identified and fixed:

- Replaced the unavailable `llama-3.1-8b-instant` model with `qwen/qwen3.6-27b`.
- Fixed Qwen router output parsing when responses contained `<think>` blocks.
- Improved the RAG prompt to match facts by meaning instead of exact wording.
- Fixed source evaluation mismatches by normalizing document metadata to filenames.
- Added Streamlit Cloud secrets support for deployment.

📁 Project Structure
```
supportpilot-ai/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
│
├── data/
│   └── sample_docs/
│       ├── knowledge_base.txt
│       ├── mini.txt
│       ├── return_policy.txt
│       └── shipping_policy.txt
│
├── rag/
│   ├── embeddings.py
│   ├── generator.py
│   ├── loader.py
│   ├── retriever.py
│   └── splitter.py
│
├── routing/
│   └── model_router.py
│
├── monitoring/
│   ├── cost_tracker.py
│   └── logger.py
│
└── evaluation/
    ├── evaluate.py
    └── test_cases.json
```
    
## 🜲 Future Scope

- Hybrid keyword + semantic retrieval
- Reranking retrieved documents
- Conversation memory
- Response caching
- Provider fallback and improved reliability
- PostgreSQL based production logging
- Improved multi document knowledge base management

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/ANJALICHAMOLI/supportpilot-cost-aware-rag.git
cd supportpilot-cost-aware-rag
```
### 2. Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Configure API keys
```
Create a .env file:

GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

```

### 5. Run the application
```
streamlit run app.py

```
➣ Upload documents from: data/sample_docs/ and start asking questions.

## ⟡ Project Highlights

- Retrieval Augmented Generation with persistent vector storage
- LLM based query routing
- Cost aware model selection
- Budget based graceful degradation
- Local embedding inference
- Request level observability
- Automated RAG evaluation
- Hallucination and edge case testing

