FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_frontend.txt .
RUN pip install --no-cache-dir -r requirements_frontend.txt

# Copy application code
COPY . .

EXPOSE 7860

# Start Chainlit server
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
