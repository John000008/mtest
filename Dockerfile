FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port from environment variable or default to 8000
EXPOSE ${PORT:-8000}

# Run the application with uvicorn
# --workers 1: Run with a single worker process (simpler for most applications)
# sh -c: Allows us to use shell variable expansion for the PORT environment variable
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
