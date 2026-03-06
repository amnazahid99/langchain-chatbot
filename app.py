from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


# ── Load Environment Configuration ───────────────────────────────────────────
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MAX_TURNS = int(os.getenv("MAX_TURNS", 5))
NUM_CTX = int(os.getenv("NUM_CTX", 5200))


# ── Model Setup ──────────────────────────────────────────────────────────────
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    num_ctx=NUM_CTX
)


# ── Prompt Template ──────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant and an expert in AI. "
     "Answer the user's questions concisely and informatively."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()


# ── Token Budget Constants ───────────────────────────────────────────────────
RESPONSE_RESERVE = 1200
SYSTEM_OVERHEAD = 60
MAX_INPUT_TOKENS = NUM_CTX - RESPONSE_RESERVE
CHARS_PER_TOKEN = 4


# ── Conversation State ───────────────────────────────────────────────────────
chat_history: list = []


# ── Token Helpers ────────────────────────────────────────────────────────────
def _tok(text: str) -> int:
    """Estimate token count from character length."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def history_token_count() -> int:
    """Estimated token count for entire chat history."""
    return sum(_tok(m.content) for m in chat_history)


def context_usage_pct() -> float:
    """Return percentage of context window used."""
    return min(100.0, history_token_count() / MAX_INPUT_TOKENS * 100)


def _fits_in_budget(question: str) -> bool:
    """Check if question alone fits inside token budget."""
    return (_tok(question) + SYSTEM_OVERHEAD) <= MAX_INPUT_TOKENS


def _trim_history(question: str) -> bool:
    """
    Remove oldest conversation pairs until the context fits budget.
    """
    q_tokens = _tok(question)
    available = MAX_INPUT_TOKENS - SYSTEM_OVERHEAD - q_tokens

    if available <= 0:
        return False

    trimmed = False

    while chat_history and history_token_count() > available:
        if len(chat_history) >= 2:
            chat_history.pop(0)
            chat_history.pop(0)
        else:
            chat_history.clear()

        trimmed = True

    return trimmed


# ── Turn Limit Enforcement ───────────────────────────────────────────────────
def _enforce_turn_limit():
    """
    Ensure conversation does not exceed MAX_TURNS.
    """
    while len(chat_history) // 2 > MAX_TURNS:
        chat_history.pop(0)
        chat_history.pop(0)


# ── Core Chat Function ───────────────────────────────────────────────────────
def chat(question: str) -> dict:
    """
    Send a question to the LLM.

    Returns:
    {
        "response": str | None,
        "q_tokens": int,
        "r_tokens": int,
        "trimmed": bool,
        "warning": str | None
    }
    """

    q_tokens = _tok(question)

    # ── Guard: Question Too Large ────────────────────────────────────────────
    if not _fits_in_budget(question):
        return {
            "response": None,
            "q_tokens": q_tokens,
            "r_tokens": 0,
            "trimmed": False,
            "warning": (
                f"Your question is too long (~{q_tokens} tokens estimated). "
                f"Maximum allowed input is {MAX_INPUT_TOKENS - SYSTEM_OVERHEAD} tokens."
            ),
        }

    # ── Trim Context If Needed ───────────────────────────────────────────────
    trimmed = _trim_history(question)

    # ── Call LLM ─────────────────────────────────────────────────────────────
    response: str = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    # ── Update History ───────────────────────────────────────────────────────
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    # ── Enforce Turn Limit ───────────────────────────────────────────────────
    _enforce_turn_limit()

    r_tokens = _tok(response)

    # ── Context Warning ──────────────────────────────────────────────────────
    usage = context_usage_pct()
    warning = None

    if usage >= 90:
        warning = (
            f"Context window is {usage:.0f}% full. "
            "Older messages will be trimmed automatically."
        )
    elif usage >= 75:
        warning = f"Context window is {usage:.0f}% full."

    return {
        "response": response,
        "q_tokens": q_tokens,
        "r_tokens": r_tokens,
        "trimmed": trimmed,
        "warning": warning,
    }


# ── CLI Smoke Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\nAI Chatbot Ready\n")

    while True:

        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            break

        result = chat(question)

        if result["response"] is None:
            print("\nWarning:", result["warning"])
            continue

        print("\nAI:", result["response"])

        print(
            f"\nTokens → Q:{result['q_tokens']}  "
            f"R:{result['r_tokens']}  "
            f"History:{history_token_count()}  "
            f"Usage:{context_usage_pct():.1f}%"
        )

        if result["warning"]:
            print("Notice:", result["warning"])