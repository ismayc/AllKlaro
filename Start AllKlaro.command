#!/bin/zsh
# Double-clickable launcher: starts the translator server and opens the page.
cd "$(dirname "$0")"

if ! curl -s -o /dev/null http://127.0.0.1:11434/api/tags; then
  echo "Starting Ollama for AllKlaro..."
  open -a Ollama 2>/dev/null || (ollama serve &> /dev/null &)
  sleep 3
fi

( sleep 2 && open "http://127.0.0.1:8710" ) &
exec uv run uvicorn server:app --host 127.0.0.1 --port 8710
