
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

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

