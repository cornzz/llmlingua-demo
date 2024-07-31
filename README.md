# LLMLingua-2 Prompt Compression Demo

## Installation
- Install Python
- Create and activate venv and install requirements:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
- Create `.env` file:
```
LLM_ENDPOINT=https://llm.example.com/v1/chat/completions
LLM_TOKEN=qwerty12345
FLAG_PASSWORD=very_secret
```

## Deployment
```
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 80 --log-level warning
```

## Development
```
source venv/bin/activate
uvicorn app:app --reload --log-level warning
```

## Inspecting flagged data
Navigate to `/flagged` and enter the password set in `.env`