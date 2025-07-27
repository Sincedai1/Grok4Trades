#!/bin/bash

# QuantumSol Setup Script
# This script helps set up the QuantumSol trading system

echo "🚀 Welcome to QuantumSol Setup 🚀"
echo "This script will guide you through the initial setup process."
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install it."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔧 Creating .env file from template..."
    cp .env.example .env
    echo "   Please edit the .env file with your configuration"
    echo "   You can use: nano .env"
    echo ""
    read -p "Press Enter to open the editor, or Ctrl+C to cancel..."
    
    # Open default editor if available
    if command -v nano &> /dev/null; then
        nano .env
    elif command -v vim &> /dev/null; then
        vim .env
    elif command -v vi &> /dev/null; then
        vi .env
    else
        echo "   No text editor found. Please edit .env manually."
    fi
else
    echo "✅ .env file already exists. Skipping creation."
fi

# Create required directories
echo ""
echo "📂 Creating required directories..."
mkdir -p ./freqtrade/user_data/strategies
mkdir -p ./hummingbot/conf/strategies
mkdir -p ./monitoring/grafana/provisioning/dashboards
mkdir -p ./monitoring/grafana/provisioning/datasources
mkdir -p ./monitoring/prometheus

echo ""
echo "🔑 Generating API keys configuration..."
# Create API keys file if it doesn't exist
if [ ! -f ./secrets/api_keys.json ]; then
    mkdir -p ./secrets
    echo '{
  "exchanges": {
    "kraken": {
      "api_key": "YOUR_KRAKEN_API_KEY",
      "api_secret": "YOUR_KRAKEN_API_SECRET"
    },
    "binance": {
      "api_key": "YOUR_BINANCE_API_KEY",
      "api_secret": "YOUR_BINANCE_API_SECRET"
    }
  },
  "solana": {
    "rpc_url": "https://api.mainnet-beta.solana.com",
    "ws_url": "wss://api.mainnet-beta.solana.com",
    "wallet_private_key": "YOUR_PHANTOM_WALLET_PRIVATE_KEY"
  },
  "telegram": {
    "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "chat_id": "YOUR_TELEGRAM_CHAT_ID"
  }
}' > ./secrets/api_keys.json
    echo "   Created ./secrets/api_keys.json - Please update with your API keys"
else
    echo "✅ API keys file already exists. Skipping creation."
fi

# Set permissions
echo ""
echo "🔒 Setting file permissions..."
chmod 600 ./.env
chmod 600 ./secrets/*
chmod +x ./scripts/*.sh 2>/dev/null

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and edit the .env file with your configuration"
echo "2. Add your API keys to ./secrets/api_keys.json"
echo "3. Start the system with: docker-compose up -d"
echo ""
echo "🌐 Access the UI at: http://localhost:8501"
echo "📊 Access Grafana at: http://localhost:3000 (admin/admin)"
echo "📈 Access Prometheus at: http://localhost:9090"
echo ""
echo "For more information, refer to the README.md file."
