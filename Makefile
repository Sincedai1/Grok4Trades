.PHONY: help test lint format install clean

help:
	@echo "Available commands:"
	@echo "  make install    Install dependencies"
	@echo "  make test       Run tests"
	@echo "  make lint       Run linters"
	@echo "  make format     Format code"
	@echo "  make clean      Clean cache files"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	flake8 core/ app/
	mypy core/ app/
	bandit -r core/ app/

format:
	black core/ app/
	isort core/ app/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/
