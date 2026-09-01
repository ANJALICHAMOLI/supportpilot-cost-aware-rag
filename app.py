

import os
import time
import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_all_documents
from rag.splitter import split_documents
from rag.embeddings import build_vector_store, load_vector_store, vector_store_exists
from rag.retriever import retrieve_relevant_chunks
from rag.generator import generate_answer
from routing.model_router import route_question
from monitoring.cost_tracker import BudgetTracker, estimate_cost
from monitoring.logger import log_request, load_logs
from evaluation.evaluate import run_evaluation


load_dotenv()

for _env_key in ("GROQ_API_KEY", "GOOGLE_API_KEY"):
    if not os.getenv(_env_key):
        try:
            if _env_key in st.secrets:
                os.environ[_env_key] = st.secrets[_env_key]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {question, answer, sources, model, ...}

if "budget_tracker" not in st.session_state:
    st.session_state.budget_tracker = BudgetTracker(budget_limit=1.0)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store() if vector_store_exists() else None


st.set_page_config(page_title="SupportPilot AI", layout="wide")
st.title("🛠️ SupportPilot AI — Cost Aware RAG Support Copilot")

tab_chat, tab_upload, tab_dashboard, tab_eval, tab_budget = st.tabs([
    "💬 Support Copilot", "📄 Document Upload", "📊 Cost & Usage Dashboard",
    "🧪 Evaluation", "💰 Budget Status",
])


# ---------------------------------------------------------------------------
# TAB: DOCUMENT UPLOAD  (Feature 1)
# ---------------------------------------------------------------------------
with tab_upload:
    st.header("Upload Support Documents")
    st.caption("Supported formats: PDF, TXT, Markdown")

    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents", disabled=not uploaded_files):
        
        os.makedirs("./uploaded_docs", exist_ok=True)
        saved_paths = []
        for uploaded_file in uploaded_files:
            save_path = os.path.join("./uploaded_docs", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(save_path)

        with st.spinner("Loading documents..."):
            documents = load_all_documents(saved_paths)

        with st.spinner("Splitting into chunks..."):
            chunks = split_documents(documents)

        with st.spinner(f"Embedding {len(chunks)} chunks and building vector store... (runs locally on CPU, no API call, first run downloads a small model)"):
            vector_store = build_vector_store(chunks)
            st.session_state.vector_store = vector_store

        st.success(f"Processed {len(saved_paths)} file(s) into {len(chunks)} chunks. Ready for questions!")

    if st.session_state.vector_store is not None:
        st.info("✅ A knowledge base is currently loaded and ready for questions.")
    else:
        st.warning("No documents loaded yet. Upload files above to get started.")


# ---------------------------------------------------------------------------
# TAB: CHAT / SUPPORT COPILOT  (Features 2, 3, 5)
# ---------------------------------------------------------------------------
with tab_chat:
    st.header("Ask a Question")

    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process documents first (see 'Document Upload' tab).")
    else:
        question = st.text_input("Your question:", key="chat_question_input")

        if st.button("Ask", disabled=not question):
            start_time = time.time()

            chunks = retrieve_relevant_chunks(st.session_state.vector_store, question, k=4)
            context_preview = "\n".join(c.page_content for c in chunks)

            routing_decision = route_question(question)

            budget_decision = st.session_state.budget_tracker.check_and_resolve_model(
                requested_model=routing_decision["model"],
                question=question,
                context_text=context_preview,
            )
            final_model = budget_decision["final_model"]

            try:
                gen_result = generate_answer(question, chunks, final_model)
                success = True
            except Exception as e:
                gen_result = {"answer": f"⚠️ Error generating answer: {e}", "sources": [], "input_tokens": 0, "output_tokens": 0}
                success = False

            latency = time.time() - start_time
            actual_cost = estimate_cost(final_model, gen_result["input_tokens"], gen_result["output_tokens"]) if success else 0.0
            st.session_state.budget_tracker.record_spend(actual_cost)

            # STEP 6 - LOGGING
            log_request(
                question=question,
                selected_model=final_model,
                routing_reason=routing_decision["reason"] + (" [BUDGET DOWNGRADE APPLIED]" if budget_decision["was_downgraded"] else ""),
                input_tokens=gen_result["input_tokens"],
                output_tokens=gen_result["output_tokens"],
                estimated_cost=actual_cost,
                latency_seconds=latency,
                success=success,
                was_budget_downgraded=budget_decision["was_downgraded"],
            )

            st.session_state.chat_history.append({
                "question": question,
                "answer": gen_result["answer"],
                "sources": gen_result["sources"],
                "chunks": chunks,
                "model": final_model,
                "routing_reason": routing_decision["reason"],
                "was_downgraded": budget_decision["was_downgraded"],
                "cost": actual_cost,
                "latency": latency,
            })

        
        for turn in reversed(st.session_state.chat_history):
            st.markdown(f"**Q: {turn['question']}**")
            st.write(turn["answer"])

           
            downgrade_note = " 🔻 (downgraded due to budget limit)" if turn["was_downgraded"] else ""
            st.caption(
                f"Model: `{turn['model']}`{downgrade_note} | "
                f"Reason: {turn['routing_reason']} | "
                f"Cost: ${turn['cost']:.6f} | Latency: {turn['latency']:.2f}s"
            )


            with st.expander(f"📚 Retrieved sources ({len(turn['sources'])})"):
                for i, chunk in enumerate(turn["chunks"]):
                    st.markdown(f"**Chunk {i+1}** — source: `{chunk.metadata.get('source', 'unknown')}`")
                    st.text(chunk.page_content)
            st.divider()


# ---------------------------------------------------------------------------
# TAB: COST & USAGE DASHBOARD  (Feature 4)
# ---------------------------------------------------------------------------
with tab_dashboard:
    st.header("Cost & Usage Dashboard")
    logs_df = load_logs()

    if logs_df.empty:
        st.info("No requests logged yet. Ask a question in the Support Copilot tab first.")
    else:

        total_queries = len(logs_df)
        total_cost = logs_df["estimated_cost"].sum()
        avg_cost = logs_df["estimated_cost"].mean()
        avg_latency = logs_df["latency_seconds"].mean()
        from routing.model_router import CHEAP_MODEL, STRONG_MODEL
        cheap_pct = (logs_df["selected_model"] == CHEAP_MODEL).mean() * 100
        strong_pct = (logs_df["selected_model"] == STRONG_MODEL).mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", total_queries)
        col2.metric("Total Estimated Cost", f"${total_cost:.4f}")
        col3.metric("Avg Cost / Request", f"${avg_cost:.6f}")
        col4.metric("Avg Latency", f"{avg_latency:.2f}s")

        col5, col6 = st.columns(2)
        col5.metric("Cheap Model Usage", f"{cheap_pct:.1f}%")
        col6.metric("Strong Model Usage", f"{strong_pct:.1f}%")

        st.subheader("Cost by Model")
        st.bar_chart(logs_df.groupby("selected_model")["estimated_cost"].sum())

        st.subheader("Average Latency by Model")
        st.bar_chart(logs_df.groupby("selected_model")["latency_seconds"].mean())

        st.subheader("Raw Request Log")
        st.dataframe(logs_df.sort_values("timestamp", ascending=False))


# ---------------------------------------------------------------------------
# TAB: EVALUATION  (Feature 6)
# ---------------------------------------------------------------------------
with tab_eval:
    st.header("Evaluation Harness")
    st.caption(
        "Runs a fixed set of ~25 test questions (evaluation/test_cases.json) through "
        "the real RAG pipeline and scores answer correctness, hallucination "
        "resistance, and retrieval accuracy. This costs real (small) API spend "
        "since it makes real LLM calls."
    )

    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process documents first.")
    else:
        if st.button("Run Evaluation"):
            with st.spinner("Running evaluation test cases through the RAG pipeline..."):
                results, summary = run_evaluation(st.session_state.vector_store)
                st.session_state.eval_results = results
                st.session_state.eval_summary = summary

        if "eval_summary" in st.session_state:
            summary = st.session_state.eval_summary
            col1, col2, col3 = st.columns(3)
            col1.metric("Pipeline Success Rate", f"{summary['pipeline_success_rate']*100:.1f}%")
            col2.metric(
                "Keyword Answer Accuracy",
                f"{summary['keyword_accuracy']*100:.1f}%" if summary["keyword_accuracy"] is not None else "N/A",
                help=f"Evaluated over {summary['keyword_cases_evaluated']} applicable cases",
            )
            col3.metric(
                "Source Retrieval Accuracy",
                f"{summary['source_retrieval_accuracy']*100:.1f}%" if summary["source_retrieval_accuracy"] is not None else "N/A",
                help=f"Evaluated over {summary['source_cases_evaluated']} applicable cases",
            )

            st.subheader("Per-Case Results")
            import pandas as pd
            results_df = pd.DataFrame(st.session_state.eval_results)
            st.dataframe(results_df)


# ---------------------------------------------------------------------------
# TAB: BUDGET STATUS  (Feature 5)
# ---------------------------------------------------------------------------
with tab_budget:
    st.header("Budget Status")

 
    new_limit = st.number_input(
        "Session budget limit ($)",
        min_value=0.0,
        value=st.session_state.budget_tracker.budget_limit,
        step=0.1,
    )
    if new_limit != st.session_state.budget_tracker.budget_limit:
        st.session_state.budget_tracker.budget_limit = new_limit

    spent = st.session_state.budget_tracker.total_spent
    remaining = st.session_state.budget_tracker.remaining_budget()

    col1, col2 = st.columns(2)
    col1.metric("Total Spent (this session)", f"${spent:.6f}")
    col2.metric("Remaining Budget", f"${remaining:.6f}")

    st.progress(min(1.0, spent / new_limit) if new_limit > 0 else 1.0)

    if remaining <= 0:
        st.error("🚫 Budget exhausted. All new questions will be routed to the cheap model regardless of complexity.")
    else:
        st.success("✅ Budget available. Complex questions will use the strong model as normal.")
