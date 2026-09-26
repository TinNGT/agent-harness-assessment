FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY data ./data
COPY README.md ./

RUN uv pip install --system .

ENV LLM_PROVIDER=scripted \
    DATABASE_URL=sqlite:////app/harness.db \
    DATA_DIR=/app/data

EXPOSE 8000

CMD ["uvicorn", "harness.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
