from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException
from googletrans import Translator
import emoji
from datetime import datetime

analyzer = SentimentIntensityAnalyzer()
translator = Translator()

def analyze_sentiment(comment):
    text = comment['text']
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'en'

    if lang != 'en':
        text = translator.translate(text, dest='en').text

    scores = analyzer.polarity_scores(text)
    return {
        'comment': comment['text'],
        'translated': text if lang != 'en' else None,
        'language': lang,
        'date': comment['date'],
        'likes': comment['likes'],
        'compound': scores['compound'],
        'positive': scores['pos'],
        'neutral': scores['neu'],
        'negative': scores['neg'],
        'sentiment': get_sentiment_label(scores['compound'])
    }

def analyze_text_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return {
        'compound': scores['compound'],
        'sentiment': get_sentiment_label(scores['compound'])
    }

def get_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_emojis(comments):
    emoji_count = {}
    for comment in comments:
        emojis = [c for c in comment['text'] if c in emoji.UNICODE_EMOJI['en']]
        for e in emojis:
            emoji_count[e] = emoji_count.get(e, 0) + 1
    return sorted(emoji_count.items(), key=lambda x: x[1], reverse=True)[:10]

def analyze_sentiment_trend(sentiment_results):
    dates = [datetime.fromisoformat(result['date'].replace('Z', '+00:00')) for result in sentiment_results]
    compound_scores = [result['compound'] for result in sentiment_results]
    
    # Sort dates and scores
    sorted_data = sorted(zip(dates, compound_scores))
    dates, compound_scores = zip(*sorted_data)
    
    # Calculate moving average
    window_size = min(len(dates) // 10, 50)  # 10% of total comments or max 50