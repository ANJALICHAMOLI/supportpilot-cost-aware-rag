
import json
import time
from rag.retriever import retrieve_relevant_chunks
from rag.generator import generate_answer
from routing.model_router import route_question


def load_test_cases(path: str = "evaluation/test_cases.json"):
   
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_keyword_match(answer: str, expected_keywords: list[str]) -> bool:
    
    if not expected_keywords:
        return None  
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def check_source_match(sources: list[str], expected_source: str) -> bool:
    
    if expected_source is None:
        return None  
    return expected_source in sources


def run_evaluation(vector_store, test_cases_path: str = "evaluation/test_cases.json", k: int = 4):
    
    test_cases = load_test_cases(test_cases_path)
    results = []

    for case in test_cases:
        question = case["question"]
        start_time = time.time()

        try:
         
            routing_decision = route_question(question)
            model_name = routing_decision["model"]

            chunks = retrieve_relevant_chunks(vector_store, question, k=k) if question.strip() else []
            gen_result = generate_answer(question, chunks, model_name) if question.strip() else {
                "answer": "I don't have information about that in the provided documents.",
                "sources": [],
                "input_tokens": 0,
                "output_tokens": 0,
            }
            success = True
            error_message = None
        except Exception as e:
      
            gen_result = {"answer": "", "sources": [], "input_tokens": 0, "output_tokens": 0}
            success = False
            error_message = str(e)

        latency = time.time() - start_time

        keyword_match = check_keyword_match(gen_result["answer"], case.get("expected_keywords", []))
        source_match = check_source_match(gen_result["sources"], case.get("expected_source"))

        results.append({
            "id": case["id"],
            "type": case["type"],
            "question": question,
            "answer": gen_result["answer"],
            "sources": gen_result["sources"],
            "expected_source": case.get("expected_source"),
            "keyword_match": keyword_match,
            "source_match": source_match,
            "should_be_answerable": case.get("should_be_answerable"),
            "latency_seconds": round(latency, 3),
            "success": success,
            "error": error_message,
        })

    summary = summarize_results(results)
    return results, summary


def summarize_results(results: list[dict]) -> dict:
   
    total = len(results)
    successful_runs = sum(1 for r in results if r["success"])

    keyword_applicable = [r for r in results if r["keyword_match"] is not None]
    keyword_correct = sum(1 for r in keyword_applicable if r["keyword_match"])

    source_applicable = [r for r in results if r["source_match"] is not None]
    source_correct = sum(1 for r in source_applicable if r["source_match"])

    avg_latency = round(sum(r["latency_seconds"] for r in results) / total, 3) if total else 0

    return {
        "total_cases": total,
        "successful_runs": successful_runs,
        "pipeline_success_rate": round(successful_runs / total, 3) if total else 0,
        "keyword_accuracy": round(keyword_correct / len(keyword_applicable), 3) if keyword_applicable else None,
        "keyword_cases_evaluated": len(keyword_applicable),
        "source_retrieval_accuracy": round(source_correct / len(source_applicable), 3) if source_applicable else None,
        "source_cases_evaluated": len(source_applicable),
        "average_latency_seconds": avg_latency,
    }
