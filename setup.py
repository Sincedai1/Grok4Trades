from setuptools import setup, find_packages

setup(
    name="grok4trades",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ccxt>=4.4.96,<5.0.0",
        "web3>=5.31.1,<7.0.0",
        "eth-account>=0.5.9,<1.0.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "requests>=2.31.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "prometheus-client",
        "uvicorn",
        "fastapi",
        "opentelemetry-instrumentation-fastapi",
        "opentelemetry-instrumentation-logging",
    ],
    python_requires=">=3.11",
)
