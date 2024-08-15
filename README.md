# LLMLingua-2 Prompt Compression Demo
Dependencies: `gradio`, `llmlingua`, `python-dotenv`

## Installation
- Install Python
- Create and activate a virtual environment and install the requirements:
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
- Create a `.env` file, e.g.:
```
LLM_ENDPOINT=https://api.openai.com/v1
LLM_TOKEN=token_1234
LLM_LIST=gpt-4o-mini, gpt-3.5-turbo     # Optional. If not provided, a list of models will be fetched from the API
FLAG_PASSWORD=very_secret               # Optional. If not provided, /flagged and /logs endpoints are disabled
```

## Running
```
source venv/bin/activate
uvicorn src.app:app --host 0.0.0.0 --port 80 --log-level warning
```
The demo is now reachable under http://localhost

## Development
```
source venv/bin/activate
uvicorn src.app:app --reload --log-level warning
```
The demo is now reachable under http://localhost:8000

## Inspecting flagged data and logs
Navigate to `/flagged` or `/logs` and enter the password set in `.env`

## Caches
- The compression model is cached in `~/.cache/huggingface`, the cache location can be set via `HF_HUB_CACHE`.
- The tokenizer vocabulary is cached in the operating systems' temporary file directory and can be set via `TIKTOKEN_CACHE_DIR`.