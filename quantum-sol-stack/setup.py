from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sniper-bot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Uniswap V2 Sniper Bot for automated trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sniper-bot",
    packages=find_packages(include=["sniper_bot", "sniper_bot.*"]),
    python_requires=">=3.8",
    install_requires=[
        "web3>=6.0.0",
        "eth-account>=0.11.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "requests>=2.31.0",
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "eth-tester>=0.9.0",
            "py-evm>=0.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "types-requests>=2.31.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sniper-bot=sniper_bot.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
