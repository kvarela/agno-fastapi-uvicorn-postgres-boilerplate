FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install agno fastapi uvicorn python-dotenv duckduckgo-search-python openai psycopg2-binary pgvector sqlalchemy pydantic python_multipart

COPY . .

EXPOSE 8000

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 