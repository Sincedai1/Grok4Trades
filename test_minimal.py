"""Minimal test script to verify FastAPI app works."""
import os
import time
import json
import threading
import subprocess
import requests
from prometheus_client import Counter, start_http_server
from fastapi import FastAPI

# Initialize metrics
g4t_requests_total = Counter('g4t_requests_total', 'Total requests', ['method', 'endpoint'])
g4t_errors_total = Counter('g4t_errors_total', 'Total errors', ['method', 'endpoint'])

app = FastAPI()

@app.get("/health")
async def health():
    """Health check endpoint."""
    g4t_requests_total.labels(method="GET", endpoint="/health").inc()
    return {"status": "healthy"}

def run_tests():
    """Run all tests against the API server."""
    # Start metrics server
    print("Starting metrics server...")
    try:
        start_http_server(9100)
        print("✅ Metrics server started")
    except Exception as e:
        print(f"❌ Failed to start metrics server: {e}")
        return False
    
    # Start API server
    print("Starting API server...")
    server = subprocess.Popen(
        ["uvicorn", "test_minimal:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    try:
        # Test health endpoint
        print("\nTesting health endpoint...")
        health_response = requests.get("http://localhost:8000/health")
        print(f"Health response: {health_response.json()}")
        assert health_response.status_code == 200
        
        # Test metrics
        print("\nTesting metrics endpoint...")
        metrics_response = requests.get("http://localhost:9100/metrics")
        print(f"Metrics response status: {metrics_response.status_code}")
        assert metrics_response.status_code == 200
        
        metrics_text = metrics_response.text
        print(f"\nMetrics content:\n{metrics_text}")
        
        assert "g4t_requests_total" in metrics_text
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        return False
        
    finally:
        print("\nShutting down servers...")
        server.terminate()
        server.wait()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
