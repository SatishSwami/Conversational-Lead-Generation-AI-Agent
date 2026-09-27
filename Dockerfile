FROM python:3.11-slim

# Prevent Python from creating .pyc files and buffer logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-root application user.
RUN useradd \
    --create-home \
    --shell /usr/sbin/nologin \
    appuser

# Install dependencies first for better Docker layer caching.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY agent ./agent
COPY core ./core
COPY database ./database
COPY tools ./tools
COPY utils ./utils
COPY autostream_kb.json .
COPY main.py .
COPY webhook_server.py .

# Runtime directories.
RUN mkdir -p /app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "webhook_server:app", "--host", "0.0.0.0", "--port", "8000"]