# LLMLingua-2 Prompt Compression Demo

## Installation
- Install Python
- Create and activate venv and install requirements:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Deployment
```
uvicorn app:app --host 0.0.0.0 --port 80 --log-level critical
```

## Development
```
uvicorn app:app --reload --log-level critical
```