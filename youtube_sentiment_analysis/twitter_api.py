import tweepy

def fetch_tweets(query, count=100):
    # Twitter API credentials
    consumer_key = 'YOUR_CONSUMER_KEY'
    consumer_secret = 'YOUR_CONSUMER_SECRET'
    access_token = 'YOUR_ACCESS_TOKEN'
    access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

    # Authenticate to Twitter
    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
    api = tweepy.API(auth)

    # Fetch tweets
    tweets = api.search(q=query, count=count, tweet_mode='extended')
    return [{'text': tweet.full_text, 'date': tweet.created_at} for tweet in tweets]