#!/usr/bin/env python3
"""
Run a backtest with the QuantumSol backtesting engine.

This script demonstrates how to run a backtest with the example strategy
and generate visualizations of the results.
"""
import os
import sys
from datetime import datetime, timedelta
from .backtest_engine import BacktestEngine, BacktestConfig, example_strategy

def main():
    # Configure the backtest
    config = BacktestConfig(
        initial_capital=10000.0,
        quote_currency='USD',
        start_date=(datetime.utcnow() - timedelta(days=365)).strftime('%Y-%m-%d'),
        end_date=datetime.utcnow().strftime('%Y-%m-%d'),
        fee_rate=0.001,  # 0.1% fee per trade
        slippage=0.0005,  # 0.05% slippage
        max_open_positions=5,
        risk_per_trade=0.01  # 1% risk per trade
    )

    # Symbols to trade (example: Bitcoin and Ethereum)
    symbols = ['BTC/USD', 'ETH/USD']
    
    print("🚀 Starting backtest...")
    print(f"Date range: {config.start_date} to {config.end_date}")
    print(f"Initial capital: ${config.initial_capital:,.2f}")
    print(f"Trading pairs: {', '.join(symbols)}")
    
    try:
        # Initialize the backtest engine with our example strategy
        engine = BacktestEngine(strategy=example_strategy, config=config)
        
        # Load historical data
        print("\n📊 Loading historical data...")
        engine.load_data(symbols=symbols, interval='1d')
        
        # Run the backtest
        print("\n⚡ Running backtest...")
        results = engine.run()
        
        # Print summary
        print("\n📈 Backtest Complete!")
        overview = results.get('overview', {})
        print(f"Initial Capital: ${overview.get('initial_capital', 0):,.2f}")
        print(f"Final Portfolio Value: ${overview.get('final_value', 0):,.2f}")
        print(f"Total Return: {overview.get('total_return_pct', 0):.2f}%")
        print(f"Annualized Return: {overview.get('annualized_return_pct', 0):.2f}%")
        print(f"Sharpe Ratio: {overview.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {overview.get('max_drawdown_pct', 0):.2f}%")
        print(f"Total Trades: {overview.get('total_trades', 0)}")
        print(f"Win Rate: {overview.get('win_rate_pct', 0):.2f}%")
        
        # Generate and save visualizations
        print("\n🖼️  Generating visualizations...")
        os.makedirs('reports', exist_ok=True)
        for name, fig in results.get('visualizations', {}).items():
            if fig is not None:
                fig.write_html(f'reports/{name}.html')
        
        print("\n✅ Backtest completed successfully!")
        print("📊 View the reports in the 'reports/' directory")
        
    except Exception as e:
        print(f"\n❌ Backtest failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
