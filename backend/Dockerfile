FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git gcc && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

# Copy application code
COPY . .

EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
