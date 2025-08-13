from setuptools import setup, find_packages

# Core dependencies that are always needed
CORE_DEPS = [
    'fastapi>=0.100.0',
    'uvicorn>=0.23.0',
    'prometheus-client>=0.17.0',
    'python-dotenv>=1.0.0',
    'loguru>=0.7.0',
]

# Optional dependencies for specific features
EXTRA_DEPS = {
    'test': ['pytest>=7.4.0', 'pytest-cov>=4.1.0', 'httpx>=0.24.0'],
    'monitor': ['opentelemetry-api>=1.19.0', 'opentelemetry-sdk>=1.19.0'],
    'all': ['ccxt>=4.4.96,<5.0.0', 'web3>=5.31.1,<7.0.0', 'eth-account>=0.5.9,<1.0.0'],
}

setup(
    name="grok4trades",
    version="0.1.0",
    packages=find_packages(),
    install_requires=CORE_DEPS,
    extras_require=EXTRA_DEPS,
    python_requires=">=3.11,<3.13",  # Explicitly support Python 3.11-3.12
)
