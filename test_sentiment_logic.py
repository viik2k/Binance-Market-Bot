import importlib.util
spec = importlib.util.spec_from_file_location("news_analysis", "news-analysis.py")
news_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(news_analysis)
import os

print("Testing Sentiment Logic...")

# Test 1: VADER (Default)
print("\nTesting VADER...")
text = "Bitcoin is crashing and the market is in fear."
print(f"Text: {text}")

# Test 2: Transformer
print("\nTesting Transformer...")
# We need to mock the pipeline to avoid downloading 500MB model if we just want to verify logic flow,
# BUT the user wants to use it. Let's try to run it. 
# If it downloads, it downloads.
try:
    score = news_analysis.analyze_transformer(text)
    print(f"Transformer Score: {score}")
except Exception as e:
    print(f"Transformer Failed: {e}")

# Test 3: OpenAI
print("\nTesting OpenAI...")
# We don't have a key, so it should return 0.0 and print error
news_analysis.OPENAI_API_KEY = None # Ensure it's empty
score = news_analysis.analyze_openai(text)
print(f"OpenAI Score (Expect 0.0): {score}")

print("\nDone.")
