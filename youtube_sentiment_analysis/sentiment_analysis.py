def analyze_social_media_sentiment(tweets, reddit_posts):
    all_comments = tweets + reddit_posts  # Combine comments from both platforms
    sentiment_results = []

    for comment in all_comments:
        sentiment = analyze_sentiment(comment['text'])
        sentiment_results.append({
            'text': comment['text'],
            'date': comment['date'],
            'sentiment': sentiment
        })

    return sentiment_results