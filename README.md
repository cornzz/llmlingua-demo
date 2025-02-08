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
LLM_ENDPOINT=https://api.openai.com/v1  # Optional. If not provided, only compression will be possible.
LLM_TOKEN=token_1234
LLM_LIST=gpt-4o-mini, gpt-3.5-turbo     # Optional. If not provided, a list of models will be fetched from the API.
SURVEY_MODE=false                       # Optional. If set to 1, survey mode is enabled, i.e. answers are returned in random order and feedback can be submitted by the user.
FLAG_PASSWORD=very_secret               # Optional. If not provided, /flagged and /logs endpoints are disabled.
APP_PATH=/compress                      # Optional. Sets the root path of the application, for example if the application is behind a reverse proxy. Do not set if the application is running on the root path.
```

## Running
```
source venv/bin/activate
uvicorn src.app:app --host 0.0.0.0 --port 80 --log-level warning
```
The demo is now reachable under http://localhost

**OR** run the demo from a docker container:

```
docker pull ghcr.io/cornzz/llmlingua-demo:main
docker run -d -e LLM_ENDPOINT=https://api.openai.com/v1 -e LLM_TOKEN=token_1234 -e LLM_LIST="gpt-4o-mini, gpt-3.5-turbo" -e FLAG_PASSWORD=very_secret -p 8000:8000 ghcr.io/cornzz/llmlingua-demo:main
```
The demo is now reachable under http://localhost:8000

> [!NOTE]  
> If you are not on a `linux/amd64` compatible platform, add `--platform linux/amd64` to the `docker pull` command to force download the image. Note that performance will be worse than if you follow the above instructions. MPS is not supported in docker containers.

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
