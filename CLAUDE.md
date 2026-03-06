# CLAUDE.md - Developer Guidance

## Project Overview

Langchain Chatbot is a conversational AI chatbot built with LangChain and Ollama, featuring intelligent token-based context management and a Streamlit web interface.

## Tech Stack

- **LangChain** (langchain-ollama) - LLM framework with LCEL chain
- **Ollama** - Local LLM runtime
- **Streamlit** - Web UI framework
- **uv** - Package manager
- **Python 3.11+** - Required version

## Key Files

- `app.py` - Main application with chat logic, token budget management, and LCEL chain
- `src/` - Modular structure with api/, backend/, config/, frontend/, settings/
- `pyproject.toml` - Project dependencies (managed with uv)
- `.env.example` - Environment variables template

## Commands

```bash
# Install dependencies
uv sync

# Run the chatbot
streamlit run app.py

# Run tests (if any)
pytest
```

## Token Budget System

The chatbot uses a token-based context management system:

- **NUM_CTX** (5200): Total context window size
- **MAX_INPUT_TOKENS** (4000): Max tokens for input (history + question)
- **RESPONSE_RESERVE** (1200): Tokens reserved for model response
- **SYSTEM_OVERHEAD** (60): Estimated tokens for system prompt

The `_trim_history()` function automatically removes oldest Human+AI message pairs when approaching the limit.

## Architecture

The app uses LangChain's LCEL (LangChain Expression Language):

```
prompt | llm | StrOutputParser
```

- **Prompt**: ChatPromptTemplate with chat_history placeholder
- **LLM**: ChatOllama with configurable model
- **Output Parser**: StrOutputParser to extract string response

## Important Notes

- Uses `minimax-m2.5:cloud` model by default
- Automatically trims chat history when approaching token limits
- Context window warnings at 75% and 90% usage
- Questions exceeding budget are rejected with a helpful message

## Adding New Features

1. Modify `app.py` for core chat logic changes
2. Add frontend components in `src/frontend/`
3. Add backend logic in `src/backend/`
4. Update configuration in `src/config/` or `src/settings/`
5. Update this file with any new architectural patterns