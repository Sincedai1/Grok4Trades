#!/usr/bin/env python3
"""Test script to verify API server functionality."""

import subprocess
import time
import requests
import sys
import os

def test_api_server():
    """Test the API server endpoints."""
    print("🚀 Starting API server test...")
    
    # Set environment variables
    env = os.environ.copy()
    env['DRY_RUN'] = '1'
    env['ALERTS_ENABLED'] = '0'
    env['PYTHONPATH'] = os.getcwd()
    
    # Start the server
    server_process = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    try:
        # Test health endpoint
        print("\n📍 Testing /health endpoint...")
        response = requests.get('http://localhost:8000/health')
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        assert response.status_code == 200
        
        # Test metrics endpoint
        print("\n📊 Testing /metrics endpoint...")
        response = requests.get('http://localhost:9100/metrics')
        print(f"  Status: {response.status_code}")
        metrics_text = response.text
        
        # Check for required metrics
        required_metrics = ['g4t_requests_total', 'g4t_errors_total']
        for metric in required_metrics:
            if metric in metrics_text:
                print(f"  ✅ Found metric: {metric}")
            else:
                print(f"  ❌ Missing metric: {metric}")
                return False
        
        # Test SSE endpoint
        print("\n📡 Testing /stream/events endpoint...")
        response = requests.get('http://localhost:8000/stream/events', stream=True, timeout=3)
        print(f"  Status: {response.status_code}")
        
        event_count = 0
        for line in response.iter_lines():
            if line:
                print(f"  Event: {line.decode()}")
                event_count += 1
                if event_count >= 3:
                    break
        
        print(f"\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
        
    finally:
        # Stop the server
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()

if __name__ == '__main__':
    # Check if required packages are installed
    try:
        import fastapi
        import uvicorn
        import prometheus_client
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install: pip install fastapi uvicorn prometheus-client")
        sys.exit(1)
    
    success = test_api_server()
    sys.exit(0 if success else 1)
