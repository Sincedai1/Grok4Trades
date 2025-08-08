FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 trader && \
    mkdir -p /app/data /app/logs && \
    chown -R trader:trader /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app app/

# Switch to non-root user
USER trader

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "app.core.bot"]