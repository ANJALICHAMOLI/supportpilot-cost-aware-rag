
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# CHEAP_MODEL = "llama-3.1-8b-instant"
CHEAP_MODEL = "qwen/qwen3.6-27b"
STRONG_MODEL = "gemini-flash-latest"

ROUTER_MODEL = "qwen/qwen3.6-27b"

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Classify the user's customer-support question as either SIMPLE or COMPLEX.

SIMPLE:
- A direct single fact lookup
- Can be answered from one piece of information
- Examples: return period, price, email, phone number, store hours
examples:
"How many days do I have to return an item?" → SIMPLE
"What is your email address?" → SIMPLE
"What are your store hours?" → SIMPLE



COMPLEX:
- Requires comparison
- Requires combining multiple facts
- Requires reasoning across different pieces of context
- Has multiple parts
- Requires a recommendation or analysis
examples:"Compare standard and express shipping." → COMPLEX
"If my return is approved, when will I get my refund and is shipping refundable?" → COMPLEX
"Which shipping option is better for me?" → COMPLEX

Return ONLY one word:
SIMPLE
or
COMPLEX"""
    ),
    ("human", "{question}")
])


def route_question(question: str) -> dict:

    question = question.strip()

    if not question:
        return {
            "model": CHEAP_MODEL,
            "reason": "Empty question -> defaulting to cheap model."
        }

    router = ChatGroq(
        model=ROUTER_MODEL,
        temperature=0,
    )

    prompt = ROUTER_PROMPT.invoke({
        "question": question
    })

    response = router.invoke(prompt)

    decision = response.content.strip().upper()

    # Qwen may include a <think> block before its final classification.
    if "</THINK>" in decision:
        decision = decision.split("</THINK>")[-1].strip()

    if decision == "COMPLEX":
        return {
            "model": STRONG_MODEL,
            "reason": "LLM router classified the question as complex -> stronger model."
        }

    if decision == "SIMPLE":
        return {
            "model": CHEAP_MODEL,
            "reason": "LLM router classified the question as simple -> cheap model."
        }

    return {
        "model": CHEAP_MODEL,
        "reason": f"Unexpected router response ({response.content!r}) -> defaulting to cheap model."
    }

# CHEAP_MODEL = "llama-3.1-8b-instant"
# STRONG_MODEL = "gemini-flash-latest"

# COMPLEX_KEYWORDS = [
#     "compare", "comparison", "difference between",
#     "pros and cons", "advantages", "disadvantages", "summarize", "summarise",
#     "what happens if", "what if", "step by step", "walk me through",
#     "recommend", "which is better", "analyze", "analyse",
# ]
# LENGTH_THRESHOLD = 120

# TOPIC_ANCHORS = ["return", "refund", "shipping", "warranty", "cancel", "exchange", "international"]


# def route_question(question: str) -> dict:

#     question_lower = question.lower().strip()

#     if not question_lower:
#         return {"model": CHEAP_MODEL, "reason": "Empty question - defaulting to cheap model."}

#     # rule 1: explicit "complex reasoning" keyword match.
#     matched_keywords = [kw for kw in COMPLEX_KEYWORDS if kw in question_lower]
#     if matched_keywords:
#         return {
#             "model": STRONG_MODEL,
#             "reason": (
#                 f"Question contains complex reasoning language "
#                 f"({', '.join(matched_keywords[:2])}) -> routed to the stronger model."
#             ),
#         }

#     # Rule 2: question length. Long questions tend to pack in multiple subquestions or a lot of nuance/context.
#     if len(question) > LENGTH_THRESHOLD:
#         return {
#             "model": STRONG_MODEL,
#             "reason": (
#                 f"Question is long ({len(question)} characters, threshold "
#                 f"{LENGTH_THRESHOLD}) -> likely multi part, routed to the stronger model."
#             ),
#         }

#     # Rule 3: question touches multiple distinct topics.
#     matched_topics = [t for t in TOPIC_ANCHORS if t in question_lower]
#     if len(matched_topics) >= 2:
#         return {
#             "model": STRONG_MODEL,
#             "reason": (
#                 f"Question spans multiple topics ({', '.join(matched_topics)}) "
#                 f"-> likely needs cross document synthesis, routed to the stronger model."
#             ),
#         }

#     # Default: nothing above fired -> treat as a simple, single-fact lookup.
#     return {
#         "model": CHEAP_MODEL,
#         "reason": "Question looks like a simple, single fact lookup -> routed to the cheap model.",
#     }
