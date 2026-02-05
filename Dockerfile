FROM python:3.10-slim

WORKDIR /app

# dépendances système (pandas / lightgbm en ont souvent besoin)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "models/train.py"]