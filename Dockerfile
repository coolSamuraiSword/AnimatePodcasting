FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PYTHONPATH}:/app"

# Create and set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Make scripts executable
RUN chmod +x /app/animate.py /app/run_tests.py

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/logs /app/cache

# Set volume for persistent data
VOLUME ["/app/data", "/app/outputs", "/app/cache", "/app/logs"]

# Set default command
ENTRYPOINT ["./animate.py"] 