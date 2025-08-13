"""Main FastAPI application for Grok4Trades."""
import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    print("OpenTelemetry not available, continuing without instrumentation")
import time
import json

# Initialize metrics
g4t_requests_total = Counter('g4t_requests_total', 'Total requests', ['method', 'endpoint'])
g4t_errors_total = Counter('g4t_errors_total', 'Total errors', ['method', 'endpoint'])
g4t_request_duration = Histogram('g4t_request_duration_seconds', 'Request duration')

# Set environment variables for testing
DRY_RUN = os.getenv('DRY_RUN', '1') == '1'
ALERTS_ENABLED = os.getenv('ALERTS_ENABLED', '0') == '1'

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    print(f"Starting Grok4Trades API - DRY_RUN: {DRY_RUN}, ALERTS: {ALERTS_ENABLED}")
    
    # Initialize instrumentation
    if OTEL_AVAILABLE:
        LoggingInstrumentor().instrument()
    
    yield
    
    # Shutdown
    print("Shutting down Grok4Trades API")

# Create FastAPI app
app = FastAPI(
    title="Grok4Trades API",
    version="0.1.0",
    lifespan=lifespan
)

# Instrument FastAPI
if OTEL_AVAILABLE:
    FastAPIInstrumentor.instrument_app(app)

@app.get("/")
async def root():
    """Root endpoint."""
    g4t_requests_total.labels(method="GET", endpoint="/").inc()
    return {"message": "Grok4Trades API", "dry_run": DRY_RUN}

@app.get("/health")
async def health():
    """Health check endpoint."""
    g4t_requests_total.labels(method="GET", endpoint="/health").inc()
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    g4t_requests_total.labels(method="GET", endpoint="/metrics").inc()
    return StreamingResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

async def event_generator():
    """Generate SSE events for testing."""
    counter = 0
    while counter < 10:
        yield f"data: {json.dumps({'event': 'test', 'counter': counter, 'timestamp': time.time()})}\n\n"
        counter += 1
        await asyncio.sleep(0.1)

@app.get("/stream/events")
async def stream_events():
    """SSE streaming endpoint for testing."""
    g4t_requests_total.labels(method="GET", endpoint="/stream/events").inc()
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# Create metrics server on port 9100
from prometheus_client import start_http_server
import threading

def start_metrics_server():
    """Start Prometheus metrics server."""
    start_http_server(9100)

# Start metrics server in background thread
try:
    print("Starting Prometheus metrics server on port 9100...")
    start_http_server(9100)
    print("✅ Metrics server started")
except Exception as e:
    print(f"❌ Failed to start metrics server: {e}")
