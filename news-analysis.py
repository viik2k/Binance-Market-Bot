
# import for environment variables and waiting
import os, time
import asyncio

# date modules that we'll most likely need
from datetime import date, datetime, timedelta

# used to grab the XML url list from a CSV file
import csv

# used to save and load coins_in_hand dictionary
import json


# modern libraries
import ccxt
import feedparser
import pytz
import sys
import lxml.html
import lxml_html_clean
lxml.html.clean = lxml_html_clean

from newspaper import Article

# Sentiment providers
from transformers import pipeline
import torch
from openai import OpenAI

# used for executing the code
from itertools import count

# we use it to time our parser execution speed
from timeit import default_timer as timer

# Use testnet (change to True) or live (change to False)?
testnet = True

# get binance key and secret from environment variables for testnet and live
api_key_test = os.getenv('binance_api_stalkbot_testnet')
api_secret_test = os.getenv('binance_secret_stalkbot_testnet')

api_key_live = os.getenv('binance_api_stalkbot_live')
api_secret_live = os.getenv('binance_secret_stalkbot_live')

# Authenticate with the client
if testnet:
    exchange = ccxt.binance({
        'apiKey': api_key_test,
        'secret': api_secret_test,
        'enableRateLimit': True,
    })
    exchange.set_sandbox_mode(True)
else:
    exchange = ccxt.binance({
        'apiKey': api_key_live,
        'secret': api_secret_live,
        'enableRateLimit': True,
    })




############################################
#     USER INPUT VARIABLES LIVE BELOW      #
# You may edit those to configure your bot #
############################################


# select what coins to look for as keywords in articles headlines
# The key of each dict MUST be the symbol used for that coin on Binance
# Use each list to define keywords separated by commas: 'XRP': ['ripple', 'xrp']
# keywords are case sensitive
keywords = {
    'XRP': ['ripple', 'xrp', 'XRP', 'Ripple', 'RIPPLE'],
    'BTC': ['BTC', 'bitcoin', 'Bitcoin', 'BITCOIN'],
    'XLM': ['Stellar Lumens', 'XLM'],
    #'BCH': ['Bitcoin Cash', 'BCH'],
    'ETH': ['ETH', 'Ethereum'],
    'BNB' : ['BNB', 'Binance Coin'],
    'LTC': ['LTC', 'Litecoin']
    }

# The Buy amount in the PAIRING symbol, by default USDT
# 100 will for example buy the equivalent of 100 USDT in Bitcoin.
QUANTITY = 100

# define what to pair each coin to
# AVOID PAIRING WITH ONE OF THE COINS USED IN KEYWORDS
PAIRING = 'USDT'

# define how positive the news should be in order to place a trade
# the number is a compound of neg, neu and pos values from the nltk analysis
# input a number between -1 and 1
SENTIMENT_THRESHOLD = 0
NEGATIVE_SENTIMENT_THRESHOLD = 0

# define the minimum number of articles that need to be analysed in order
# for the sentiment analysis to qualify for a trade signal
# avoid using 1 as that's not representative of the overall sentiment
MINUMUM_ARTICLES = 1

# define how often to run the code (check for new + try to place trades)
# in minutes
REPEAT_EVERY = 60

# define how old an article can be to be included
# in hours
# define how old an article can be to be included
# in hours
HOURS_PAST = 24

# Sentiment Provider Configuration
# Options: 'transformer', 'openai'
SENTIMENT_PROVIDER = 'transformer' 

# OpenAI Configuration (if using 'openai' provider)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = 'gpt-3.5-turbo'

# Transformer Configuration (if using 'transformer' provider)
# A good financial sentiment model
TRANSFORMER_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"


############################################
#        END OF USER INPUT VARIABLES       #
#             Edit with care               #
############################################




# coins that bought by the bot since its start
coins_in_hand  = {}

# path to the saved coins_in_hand file
coins_in_hand_file_path = 'coins_in_hand.json'

# use separate files for testnet and live
if testnet:
    coins_in_hand_file_path = 'testnet_' + coins_in_hand_file_path

# if saved coins_in_hand json file exists then load it
if os.path.isfile(coins_in_hand_file_path):
    with open(coins_in_hand_file_path) as file:
        coins_in_hand = json.load(file)

# and add coins from actual keywords if they aren't in coins_in_hand dictionary already
for coin in keywords:
    if coin not in coins_in_hand:
        coins_in_hand[coin] = 0


def get_price(symbol):
    '''Get the current price for a symbol'''
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None



# Load markets to get precision info
exchange.load_markets()

def get_lot_size(symbol):
    if symbol in exchange.markets:
        return exchange.markets[symbol]['precision']['amount']
    return None




def calculate_volume():
    volume = {}
    for coin in keywords:
        symbol = coin + PAIRING
        price = get_price(symbol)
        if price:
            # Calculate amount to buy: QUANTITY (in USDT) / Price
            amount = QUANTITY / float(price)
            # Adjust to precision
            volume[symbol] = exchange.amount_to_precision(symbol, amount)
    return volume


# load the csv file containg top 100 crypto feeds
# want to scan other websites?
# Simply add the RSS Feed url to the Crypto feeds.csv file
with open('Crypto feeds.csv') as csv_file:

    # open the file
    csv_reader = csv.reader(csv_file)

    # remove any headers
    next(csv_reader, None)

    # create empty list
    feeds = []

    # add each row cotaining RSS url to feeds list
    for row in csv_reader:
        feeds.append(row[0])

# Make headlines global variable as it should be the same across all functions
headlines = {'source': [], 'title': [], 'pubDate' : [], 'text': []}




def get_headlines():
    '''
    Fetch news from RSS feeds using feedparser and newspaper3k
    '''
    print("Fetching news...")
    # clear headlines
    headlines['source'] = []
    headlines['title'] = []
    headlines['pubDate'] = []
    headlines['text'] = [] # New field for full text/summary

    for feed in feeds:
        try:
            parsed_feed = feedparser.parse(feed)
            for entry in parsed_feed.entries:
                # Check date
                published = None
                if hasattr(entry, 'published_parsed'):
                    published = datetime.fromtimestamp(time.mktime(entry.published_parsed)).replace(tzinfo=pytz.utc)
                elif hasattr(entry, 'updated_parsed'):
                    published = datetime.fromtimestamp(time.mktime(entry.updated_parsed)).replace(tzinfo=pytz.utc)
                
                if published:
                    time_between = datetime.now(pytz.utc) - published
                    if time_between.total_seconds() / 3600 <= HOURS_PAST:
                        headlines['source'].append(feed)
                        headlines['pubDate'].append(published)
                        headlines['title'].append(entry.title)
                        
                        # Try to get full text with newspaper
                        try:
                            article = Article(entry.link)
                            article.download()
                            article.parse()
                            # article.nlp() # Optional: heavy
                            headlines['text'].append(article.text if article.text else entry.title)
                        except:
                            # Fallback to description or title
                            content = getattr(entry, 'description', entry.title)
                            headlines['text'].append(content)
                        
                        print(f"Found: {entry.title}")
        except Exception as e:
            print(f"Error parsing {feed}: {e}")




def categorise_headlines():
    '''arrange all headlines scaped in a dictionary matching the coin's name'''
    # get the headlines
    get_headlines()
    categorised_headlines = {}

    # this loop will create a dictionary for each keyword defined
    for keyword in keywords:
        categorised_headlines['{0}'.format(keyword)] = []

    # keyword needs to be a loop in order to be able to append headline to the correct dictionary
    for keyword in keywords:
        # looping through each headline is required as well
        for i, title in enumerate(headlines['title']):
            text = headlines['text'][i]
            # check keywords in title (or text?) - let's check both
            if any(key in title for key in keywords[keyword]) or any(key in text for key in keywords[keyword]):
                # append the text for sentiment analysis
                categorised_headlines[keyword].append(text)

    return categorised_headlines



def analyze_transformer(text):
    '''Analyze sentiment using HuggingFace transformers'''
    try:
        # Initialize pipeline (lazy load could be better but for simplicity here)
        # Note: This might be slow to load on first run
        classifier = pipeline('sentiment-analysis', model=TRANSFORMER_MODEL)
        result = classifier(text[:512])[0] # Truncate to 512 tokens
        
        # Map labels to compound score
        # Model specific: usually 'positive', 'negative', 'neutral'
        label = result['label'].lower()
        score = result['score']
        
        if 'negative' in label:
            return -score
        elif 'positive' in label:
            return score
        else:
            return 0.0
    except Exception as e:
        print(f"Transformer error: {e}")
        return 0.0

def analyze_openai(text):
    '''Analyze sentiment using OpenAI'''
    try:
        if not OPENAI_API_KEY:
            print("OpenAI API Key not set!")
            return 0.0
            
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial sentiment analyzer. Analyze the following news headline/text and return a single float number between -1.0 (extremely negative) and 1.0 (extremely positive). Return ONLY the number."},
                {"role": "user", "content": text}
            ]
        )
        content = response.choices[0].message.content.strip()
        return float(content)
    except Exception as e:
        print(f"OpenAI error: {e}")
        return 0.0

def analyse_headlines():
    '''Analyse categorised headlines and return NLP scores'''
    categorised_headlines = categorise_headlines()
    sentiment = {}
    
    for coin in categorised_headlines:
        if len(categorised_headlines[coin]) > 0:
            # create dict for each coin
            sentiment['{0}'.format(coin)] = []
            # append sentiment to dict
            for text in categorised_headlines[coin]:
                score = 0.0
                if SENTIMENT_PROVIDER == 'transformer':
                    score = analyze_transformer(text)
                elif SENTIMENT_PROVIDER == 'openai':
                    score = analyze_openai(text)
                
                # For consistency with old structure which expected a dict with 'compound'
                # We will just store the float score directly in our new list, 
                # BUT compile_sentiment expects a list of dicts or we need to change compile_sentiment.
                # Let's just return a list of dicts to be safe and minimize refactor of compile_sentiment
                sentiment[coin].append({'compound': score})

    return sentiment


def compile_sentiment():
    '''Arranges every compound value into a list for each coin'''
    sentiment = analyse_headlines()
    compiled_sentiment = {}

    for coin in sentiment:
        compiled_sentiment[coin] = []

        for item in sentiment[coin]:
            # append each compound value to each coin's dict
            compiled_sentiment[coin].append(sentiment[coin][sentiment[coin].index(item)]['compound'])

    return compiled_sentiment


def compound_average():
    '''Calculates and returns the average compoud sentiment for each coin'''
    compiled_sentiment = compile_sentiment()
    headlines_analysed = {}

    for coin in compiled_sentiment:
        headlines_analysed[coin] = len(compiled_sentiment[coin])

        # calculate the average using numpy if there is more than 1 element in list
        compiled_sentiment[coin] = np.array(compiled_sentiment[coin])

        # get the mean
        compiled_sentiment[coin] = np.mean(compiled_sentiment[coin])

        # convert to scalar
        compiled_sentiment[coin] = compiled_sentiment[coin].item()

    return compiled_sentiment, headlines_analysed





def buy(compiled_sentiment, headlines_analysed):
    '''Check if the sentiment is positive and keyword is found for each handle'''
    volume = calculate_volume()
    for coin in compiled_sentiment:
        # check if the sentiment and number of articles are over the given threshold
        if compiled_sentiment[coin] > SENTIMENT_THRESHOLD and headlines_analysed[coin] >= MINUMUM_ARTICLES and coins_in_hand[coin]==0:
            # check the volume looks correct
            price = get_price(coin+PAIRING)
            print(f'preparing to buy {volume[coin+PAIRING]} {coin} with {PAIRING} at {price}')

            if (testnet):
                # create test order before pushing an actual order
                pass 

            # try to create a real order
            try:
                buy_limit = exchange.create_order(
                    symbol=coin+PAIRING,
                    side='buy',
                    type='market',
                    amount=volume[coin+PAIRING]
                )
            except Exception as e:
                print(e)
            else:
                # adds coin to our portfolio
                coins_in_hand[coin] += volume[coin+PAIRING]

                # retrieve the last order
                # For simplicity, use the returned buy_limit order object
                order = [buy_limit]

                if order:
                    # convert order timsestamp into UTC format
                    time_ts = order[0]['timestamp'] / 1000
                    utc_time = datetime.fromtimestamp(time_ts)

                    # grab the price of CRYPTO the order was placed at for reporting
                    bought_at = get_price(coin+PAIRING)

                    # print order condirmation to the console
                    print(f"order {order[0]['id']} has been placed on {coin} with {order[0]['amount']} at {utc_time} and bought at {bought_at}")
                else:
                    print('Could not get last order from Binance!')

        else:
            print(f'Sentiment not positive enough for {coin}, or not enough headlines analysed or already bought: {compiled_sentiment[coin]}, {headlines_analysed[coin]}')


def sell(compiled_sentiment, headlines_analysed):
    '''Check if the sentiment is negative and keyword is found for each handle'''
    for coin in compiled_sentiment:
        # check if the sentiment and number of articles are over the given threshold
        if compiled_sentiment[coin] < NEGATIVE_SENTIMENT_THRESHOLD and headlines_analysed[coin] >= MINUMUM_ARTICLES and coins_in_hand[coin]>0:
            # check the volume looks correct
            price = get_price(coin+PAIRING)
            print(f'preparing to sell {coins_in_hand[coin]} {coin} at {price}')

            # amount_to_sell = calculate_one_volume_from_lot_size(coin+PAIRING, coins_in_hand[coin]*99.5/100)
            # Simplify amount calculation using ccxt precision
            amount_to_sell = exchange.amount_to_precision(coin+PAIRING, coins_in_hand[coin]*0.995)

            if (testnet):
                pass

            # try to create a real order
            try:
                sell_limit = exchange.create_order(
                    symbol=coin+PAIRING,
                    side='sell',
                    type='market',
                    amount=amount_to_sell
                )
            except Exception as e:
                print(e)
            else:
                # set coin to 0
                coins_in_hand[coin]=0
                # retrieve the last order
                order = [sell_limit]

                if order:
                    # convert order timsestamp into UTC format
                    time_ts = order[0]['timestamp'] / 1000
                    utc_time = datetime.fromtimestamp(time_ts)

                    # grab the price of CRYPTO the order was placed at for reporting
                    sold_at = get_price(coin+PAIRING)

                    # print order condirmation to the console
                    print(f"order {order[0]['id']} has been placed on {coin} with {order[0]['amount']} coins sold for {sold_at} each at {utc_time}")
                else:
                    print('Could not get last order from Binance!')

        else:
            print(f'Sentiment not negative enough for {coin}, not enough headlines analysed or not enough {coin} to sell: {compiled_sentiment[coin]}, {headlines_analysed[coin]}')


def save_coins_in_hand_to_file():
    # abort saving if dictionary is empty
    if not coins_in_hand:
        return

    # save coins_in_hand to file
    with open(coins_in_hand_file_path, 'w') as file:
        json.dump(coins_in_hand, file, indent=4)



if __name__ == '__main__':
    print('Press Ctrl-Q to stop the script')
    for i in count():
        compiled_sentiment, headlines_analysed = compound_average()
        print("\nBUY CHECKS:")
        buy(compiled_sentiment, headlines_analysed)
        print("\nSELL CHECKS:")
        sell(compiled_sentiment, headlines_analysed)
        print('\nCurrent bot holdings: ')
        for coin in coins_in_hand:
            if coins_in_hand[coin] > 0:
                print(f'{coin}: {coins_in_hand[coin]}')
        save_coins_in_hand_to_file()
        print(f'\nIteration {i}')
        time.sleep(60 * REPEAT_EVERY)
