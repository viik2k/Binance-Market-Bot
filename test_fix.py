import sys
import os

# Add the current directory to sys.path so we can import news-analysis
sys.path.append(os.getcwd())

# Mocking the necessary parts to avoid full execution
import importlib.util
spec = importlib.util.spec_from_file_location("news_analysis", "news-analysis.py")
news_analysis = importlib.util.module_from_spec(spec)
sys.modules["news_analysis"] = news_analysis
spec.loader.exec_module(news_analysis)

def test_analyse_headlines():
    print("Testing analyse_headlines...")
    
    # Mock headlines
    news_analysis.headlines = {
        'source': ['test_source'],
        'title': ['Bitcoin hits all time high due to ETF approval'],
        'pubDate': ['2023-01-01'],
        'text': ['Bitcoin hits all time high due to ETF approval. This is very positive news for the crypto market.']
    }
    
    # Mock get_headlines to do nothing so it doesn't overwrite our mock data
    news_analysis.get_headlines = lambda: None
    
    # Mock keywords to ensure we have a match
    news_analysis.keywords = {
        'BTC': ['Bitcoin']
    }
    
    # Run analysis
    try:
        sentiment = news_analysis.analyse_headlines()
        print(f"Sentiment result: {sentiment}")
        
        if 'BTC' in sentiment and len(sentiment['BTC']) > 0:
            score = sentiment['BTC'][0]['compound']
            print(f"BTC Score: {score}")
            if isinstance(score, float):
                print("SUCCESS: Score is a float.")
            else:
                print("FAILURE: Score is not a float.")
        else:
            print("FAILURE: No sentiment calculated for BTC.")
            
    except Exception as e:
        print(f"CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analyse_headlines()
