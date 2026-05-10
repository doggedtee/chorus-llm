FROM python:3.11-slim

WORKDIR /app

# install dependencies first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY . .

# create data directory for SQLite + research DB
RUN mkdir -p /app/data /app/db_data

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]