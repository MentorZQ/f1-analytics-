FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_core.txt .
RUN pip install --no-cache-dir -r requirements_core.txt

COPY query.py reingest.py server.py ./
COPY RAG_data_layers/ RAG_data_layers/
COPY events_and_chunking/ events_and_chunking/

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
