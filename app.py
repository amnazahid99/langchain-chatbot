from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# ── Model ─────────────────────────────────────────────────────────────────────
llm = ChatOllama(
    model="minimax-m2.5:cloud",
    temperature=0.7,
    num_ctx=5200,
)

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", "You are a helpful assistant and an expert in AI. "
               "You will answer the user's question in a concise and informative manner."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

# ── Token Budget Constants ────────────────────────────────────────────────────
NUM_CTX         = 5200   # must match llm num_ctx
RESPONSE_RESERVE = 1200  # tokens kept free for the model's reply
SYSTEM_OVERHEAD  = 60    # estimated tokens for system prompt + template wrapper
MAX_INPUT_TOKENS = NUM_CTX - RESPONSE_RESERVE   # 4 000 tokens for input
CHARS_PER_TOKEN  = 4     # 1 token ≈ 4 characters (conservative estimate)

# ── Conversation State ────────────────────────────────────────────────────────
chat_history: list = []   # list[HumanMessage | AIMessage]


# ── Token Helpers ─────────────────────────────────────────────────────────────
def _tok(text: str) -> int:
    """Estimate token count from character length."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def history_token_count() -> int:
    """Return estimated token count of the entire chat_history."""
    return sum(_tok(m.content) for m in chat_history)


def context_usage_pct() -> float:
    """Return history token usage as a percentage of MAX_INPUT_TOKENS."""
    return min(100.0, history_token_count() / MAX_INPUT_TOKENS * 100)


def _fits_in_budget(question: str) -> bool:
    """
    True if the question alone (+ overhead) leaves room inside MAX_INPUT_TOKENS.
    If False the question itself is too long to ever fit.
    """
    return (_tok(question) + SYSTEM_OVERHEAD) <= MAX_INPUT_TOKENS


def _trim_history(question: str) -> bool:
    """
    Pop the oldest Human+AI pair from chat_history until the combined
    budget (history + question + overhead) fits within MAX_INPUT_TOKENS.
    Always removes complete pairs to keep history coherent.
    Returns True if any trimming was done.
    """
    q_tokens = _tok(question)
    available = MAX_INPUT_TOKENS - SYSTEM_OVERHEAD - q_tokens
    if available <= 0:
        return False            # question alone is too big; nothing to trim

    trimmed = False
    while chat_history and history_token_count() > available:
        if len(chat_history) >= 2:
            chat_history.pop(0)   # oldest HumanMessage
            chat_history.pop(0)   # its AIMessage
        else:
            chat_history.clear()
        trimmed = True

    return trimmed


# ── Core Chat Function ────────────────────────────────────────────────────────
def chat(question: str) -> dict:
    """
    Send *question* to the LLM and return a result dict:

    {
        "response"  : str | None,   # None when question is too long
        "q_tokens"  : int,          # estimated tokens in the question
        "r_tokens"  : int,          # estimated tokens in the response
        "trimmed"   : bool,         # True if old history was pruned
        "warning"   : str | None,   # non-fatal advisory string, or None
    }
    """
    q_tokens = _tok(question)

    # ── Guard: question alone is too large ────────────────────────────────────
    if not _fits_in_budget(question):
        return {
            "response": None,
            "q_tokens": q_tokens,
            "r_tokens": 0,
            "trimmed":  False,
            "warning": (
                f"Your question is too long (~{q_tokens} tokens estimated). "
                f"The input budget is {MAX_INPUT_TOKENS - SYSTEM_OVERHEAD} tokens. "
                "Please shorten your question and try again."
            ),
        }

    # ── Trim history if needed ─────────────────────────────────────────────────
    trimmed = _trim_history(question)

    # ── Call the LLM ──────────────────────────────────────────────────────────
    response: str = chain.invoke({
        "question":     question,
        "chat_history": chat_history,
    })

    # ── Update history ────────────────────────────────────────────────────────
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    r_tokens = _tok(response)

    # ── Optional soft warning when context is getting full ───────────────────
    usage = context_usage_pct()
    warning = None
    if usage >= 90:
        warning = (
            f"Context window is {usage:.0f}% full. "
            "Older messages will be trimmed automatically on the next turn."
        )
    elif usage >= 75:
        warning = f"Context window is {usage:.0f}% full."

    return {
        "response": response,
        "q_tokens": q_tokens,
        "r_tokens": r_tokens,
        "trimmed":  trimmed,
        "warning":  warning,
    }


# ── Quick smoke-test (only runs with: python app.py) ─────────────────────────
if __name__ == "__main__":
    for q in ("What is NLP?", "What are its applications?", "What is RAG?"):
        res = chat(q)
        print(f"\nQ: {q}")
        print(f"A: {res['response']}")
        print(f"   tokens → q:{res['q_tokens']}  r:{res['r_tokens']}  "
              f"history:{history_token_count()}  usage:{context_usage_pct():.1f}%")