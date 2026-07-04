# chat_cli

A terminal-based AI assistant that runs in the command line and supports three topic domains: cooking, history, and programming.

## Features

- Topic-scoped assistant persona for `cooking`, `history`, and `programming`
- Streaming AI responses in the terminal via Rich
- Conversation memory within a session
- Commands for topic switching, history review, transcript saving, and session control
- Uses Groq via `langchain-groq` and `langchain-core`

## Requirements

- Python 3.11+ recommended
- A Groq API key
- Dependencies installed from `requirements.txt` or via `pip`

## Setup

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Groq API key:

```text
GROQ_API_KEY=gsk_...
```

## Run

```powershell
python cli_assistant.py
```

## Commands

- `/topic <cooking|history|programming>` — switch the assistant domain
- `/history` — print the full conversation so far
- `/tokens` — show turn count in the current session
- `/save` — save the conversation transcript to `transcript_*.md`
- `/clear` — clear conversation history
- `/help` — display command help
- `/exit` — quit

## Notes

- The assistant uses a fresh system prompt each turn so topic switching changes behavior immediately.
- Streaming mode does not expose per-turn token usage for Groq, so `/tokens` reports turn count only.
