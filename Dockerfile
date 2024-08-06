FROM python:3.11-slim

WORKDIR /demo
COPY . .
RUN pip install -r requirements.txt
EXPOSE 80

CMD ["uvicorn", "src.app:app", "--host 0.0.0.0", "--port 80", "--log-level warning"]
