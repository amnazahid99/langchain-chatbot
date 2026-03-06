# Langchain Chatbot

A conversational AI chatbot built with LangChain and Ollama, featuring intelligent context management and a modern web interface.

## Features

- **LangChain LCEL Chain** - Uses LangChain's Language Chain Execution Layer for flexible pipeline composition
- **Ollama Integration** - Connects to local Ollama instance for running LLM models
- **Token Budget Management** - Automatically manages conversation context to stay within model limits
- **Streamlit UI** - Clean and intuitive web interface for chatting

## Tech Stack

- **LangChain** - Framework for building LLM applications
- **Ollama** - Local LLM runtime
- **Streamlit** - Web UI framework
- **Python 3.11+** - Required Python version

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langchain-chatbot
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Ollama settings
```

4. Ensure Ollama is running with your preferred model:
```bash
ollama run minimax-m2.5:cloud
```

## Usage

Run the chatbot:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## Project Structure

```
langchain-chatbot/
├── app.py              # Main application with chat logic
├── src/
│   ├── api/            # API endpoints
│   ├── backend/        # Backend logic
│   ├── config/         # Configuration files
│   ├── frontend/       # Frontend components
│   └── settings/       # Settings management
├── pyproject.toml      # Project dependencies
└── README.md           # This file
```

## Configuration

Key parameters in `app.py`:

- **Model**: `minimax-m2.5:cloud` (configurable)
- **Temperature**: `0.7` - Controls response creativity
- **Context Window**: `5200` tokens
- **Max Input Tokens**: `4000` tokens (after reserving space for response)

## Token Management

The chatbot automatically:
- Monitors token usage in conversations
- Trims old messages when context approaches capacity
- Warns user when context window is >75% full
- Rejects questions that exceed available budget

## License

MIT