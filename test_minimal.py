"""Minimal test script to verify FastAPI app works."""
from fastapi import FastAPI
from prometheus_client import Counter

# Initialize metrics
g4t_requests_total = Counter('g4t_requests_total', 'Total requests', ['method', 'endpoint'])
g4t_errors_total = Counter('g4t_errors_total', 'Total errors', ['method', 'endpoint'])

app = FastAPI()

@app.get("/health")
async def health():
    """Health check endpoint."""
    g4t_requests_total.labels(method="GET", endpoint="/health").inc()
    return {"status": "healthy"}
