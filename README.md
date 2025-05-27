## LangGraph Stateless Chatbot

Stateless chatbot using LangGraph and OpenAI's GPT-4o. Responds to messages and returns UTC time via `get_current_time` tool.

## Setup

1. Clone repo:
   ```bash
   git clone https://github.com/ssaabbii/langgraph-chatbot.git
   cd langgraph-chatbot
   ```

2. Set up virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set OpenAI API key:
   ```bash
   cp template.env .env
   ```
   Edit `.env` with `OPENAI_API_KEY=sk-...`.

5. Run:
   ```bash
   langgraph dev
   ```

## Usage

- Send messages in the LangGraph interface.
- "What time is it?" triggers the time tool.

## Files

- `app.py`: Chatbot logic.
- `requirements.txt`: Dependencies.
- `template.env`: API key template.
- `README.md`: This file.

