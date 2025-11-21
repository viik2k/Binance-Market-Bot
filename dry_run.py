import os
import sys

# Set mock environment variables
os.environ['binance_api_stalkbot_testnet'] = 'mock_api_key'
os.environ['binance_secret_stalkbot_testnet'] = 'mock_secret_key'
os.environ['binance_api_stalkbot_live'] = 'mock_live_key'
os.environ['binance_secret_stalkbot_live'] = 'mock_live_secret'

print("Starting Dry Run...")

# Import the script - this will initialize the client
try:
    # We need to import it as a module, but it's a script with hyphens in name.
    # Renaming it temporarily or using importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("news_analysis", "news-analysis.py")
    news_analysis = importlib.util.module_from_spec(spec)
    
    # Monkey patch ccxt to avoid real network calls if needed, 
    # but we are using sandbox mode so it might be fine if keys are fake?
    # Actually, with fake keys, ccxt will error on auth.
    # We should mock the exchange object in the module.
    
    # But the module initializes exchange at top level.
    # So we need to mock ccxt before importing.
    
    import unittest.mock as mock
    import ccxt
    
    # Mock the exchange
    mock_exchange = mock.Mock()
    mock_exchange.fetch_ticker.return_value = {'last': 50000.0}
    mock_exchange.amount_to_precision.side_effect = lambda s, a: a
    mock_exchange.create_order.return_value = {
        'id': 'mock_order_id',
        'amount': 0.002,
        'timestamp': 1678886400000, # 2023-03-15
        'origQty': 0.002 # For compatibility if needed
    }
    mock_exchange.markets = {'BTC/USDT': {'precision': {'amount': 6}}}
    
    # Patch ccxt.binance to return our mock exchange
    with mock.patch('ccxt.binance', return_value=mock_exchange):
        sys.modules['ccxt'] = mock.Mock()
        sys.modules['ccxt'].binance = mock.Mock(return_value=mock_exchange)
        
        spec.loader.exec_module(news_analysis)
        
        # Now run one cycle
        print("Running one cycle...")
        compiled_sentiment, headlines_analysed = news_analysis.compound_average()
        print("Sentiment:", compiled_sentiment)
        
        print("\nBUY CHECKS:")
        news_analysis.buy(compiled_sentiment, headlines_analysed)
        
        print("\nSELL CHECKS:")
        news_analysis.sell(compiled_sentiment, headlines_analysed)
        
        print("\nDry Run Complete!")

except Exception as e:
    print(f"Crashed: {e}")
    import traceback
    traceback.print_exc()
