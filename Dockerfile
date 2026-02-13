FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8000

ENV MODEL_PATH=/app/artifacts/models/baseline_cnn.pt

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
