#!/bin/bash

# Activate the virtual environment
source /Users/sincedai1/Desktop/Grok4Trades/.venv/bin/activate

# Change to the project directory where docker-compose.yml exists
cd /Users/sincedai1/Desktop/Grok4Trades/quantum-sol-stack || { echo "Directory not found! Exiting."; exit 1; }

# Build and start the services in detached mode
docker-compose up -d --build

# Check running services
docker-compose ps

# Optionally, tail logs for specific services (uncomment as needed)
# docker-compose logs -f hummingbot
# docker-compose logs -f ui
# docker-compose logs -f agent  # If you have an 'agent' service
