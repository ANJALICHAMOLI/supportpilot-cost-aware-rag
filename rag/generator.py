
import time
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from routing.model_router import CHEAP_MODEL, STRONG_MODEL


RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful customer support assistant. Answer the user's question "
     "using ONLY the information in the context below.\n\n"
     "IMPORTANT - MATCH BY MEANING, NOT BY EXACT WORDS: the context may state a "
     "fact using different words than the question does. Identify what fact the "
     "question is actually asking for, then check whether that fact is present "
     "in the context in ANY phrasing - do not require the question's exact "
     "words to appear in the context. "
     "Example: if asked for a 'phone number' and the context says 'call "
     "555-0100', that DOES answer the question, because a phone number is what "
     "you call. Extract and state the actual value (555-0100), not a refusal.\n\n"
     "If the context contains the answer in any form, state it directly and "
     "concisely, using the exact value from the context (the actual number, "
     "date, amount, or fact).\n\n"
     "If the context genuinely does not contain information relevant to the "
     "question, respond exactly with: "
     "\"I don't have information about that in the provided documents.\"\n\n"
     "Never invent or infer facts that are not explicitly present in the "
     "context, even to fill in plausible sounding details.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

def format_context(chunks) -> str:
    
    formatted_pieces = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown source")
        formatted_pieces.append(f"[Source {i+1}: {source}]\n{chunk.page_content}")
    return "\n\n".join(formatted_pieces)


def get_llm(model_name: str, temperature: float = 0.0):
    
    if model_name == CHEAP_MODEL:
        return ChatGroq(model=model_name, temperature=temperature)
    elif model_name == STRONG_MODEL:
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Expected one of: {CHEAP_MODEL}, {STRONG_MODEL}")


def _invoke_with_retry(llm, prompt_messages, max_retries: int = 2, backoff_seconds: float = 3.0):
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(prompt_messages)
        except Exception as e:
            last_error = e
            error_text = str(e).lower()
            is_transient = any(
                token in error_text
                for token in ["429", "rate limit", "quota", "503", "timeout", "unavailable"]
            )
            if not is_transient or attempt == max_retries:
                raise
            time.sleep(backoff_seconds * (attempt + 1))  
    raise last_error 


def generate_answer(question: str, chunks, model_name: str):
    context_text = format_context(chunks)
    llm = get_llm(model_name)

    prompt_messages = RAG_PROMPT_TEMPLATE.invoke({
        "context": context_text,
        "question": question,
    })

    response = _invoke_with_retry(llm, prompt_messages)

 
    usage = response.usage_metadata or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    sources = list(dict.fromkeys(
        chunk.metadata.get("source", "unknown source") for chunk in chunks
    ))

    return {
        "answer": response.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "sources": sources,
    }
