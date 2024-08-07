FROM python:3.11-slim

WORKDIR /demo
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt

COPY . .
EXPOSE 80

CMD ["uvicorn", "src.app:app", "--host 0.0.0.0", "--port 80", "--log-level warning"]
