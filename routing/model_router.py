
CHEAP_MODEL = "llama-3.1-8b-instant"
STRONG_MODEL = "gemini-flash-latest"

COMPLEX_KEYWORDS = [
    "compare", "comparison", "difference between",
    "pros and cons", "advantages", "disadvantages", "summarize", "summarise",
    "what happens if", "what if", "step by step", "walk me through",
    "recommend", "which is better", "analyze", "analyse",
]

LENGTH_THRESHOLD = 120


TOPIC_ANCHORS = ["return", "refund", "shipping", "warranty", "cancel", "exchange", "international"]


def route_question(question: str) -> dict:
    
    question_lower = question.lower().strip()

    if not question_lower:
        
        return {"model": CHEAP_MODEL, "reason": "Empty question - defaulting to cheap model."}

    matched_keywords = [kw for kw in COMPLEX_KEYWORDS if kw in question_lower]
    if matched_keywords:
        return {
            "model": STRONG_MODEL,
            "reason": (
                f"Question contains complex-reasoning language "
                f"({', '.join(matched_keywords[:2])}) -> routed to the stronger model."
            ),
        }


    if len(question) > LENGTH_THRESHOLD:
        return {
            "model": STRONG_MODEL,
            "reason": (
                f"Question is long ({len(question)} characters, threshold "
                f"{LENGTH_THRESHOLD}) -> likely multi-part, routed to the stronger model."
            ),
        }


    matched_topics = [t for t in TOPIC_ANCHORS if t in question_lower]
    if len(matched_topics) >= 2:
        return {
            "model": STRONG_MODEL,
            "reason": (
                f"Question spans multiple topics ({', '.join(matched_topics)}) "
                f"-> likely needs cross-document synthesis, routed to the stronger model."
            ),
        }

    return {
        "model": CHEAP_MODEL,
        "reason": "Question looks like a simple, single-fact lookup -> routed to the cheap model.",
    }
