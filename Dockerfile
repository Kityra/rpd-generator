FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --uid 10001 --create-home appuser

COPY .streamlit .streamlit
COPY *.py final_schema.json ./
COPY config/ config/
COPY activity/ activity/
COPY auth/ auth/
COPY domain/ domain/
COPY orchestration/ orchestration/
COPY skills/ skills/
COPY rabota_it/ rabota_it/
COPY storage/ storage/
COPY apps/ apps/
COPY workers/ workers/
COPY scripts/migrate_sqlite_to_postgres.py scripts/
COPY alembic.ini ./
RUN mkdir -p /app/data && chown appuser:appuser /app/data

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).close()"]

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
