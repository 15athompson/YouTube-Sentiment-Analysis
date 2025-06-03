import pytest

def test_fetch_comments():
    video_id = 'VIDEO_ID'
    comments = fetch_comments(video_id)
    assert len(comments) > 0

def test_analyze_sentiment():
    comments = ['This is a positive comment', 'This is a negative comment']
    sentiments = analyze_sentiment(comments)
    assert len(sentiments) == 2

def test_store_results():
    sentiments = [{'polarity': 0.5, 'subjectivity': 0.5}, {'polarity': -0.5, 'subjectivity': 0.5}]
    store_results(sentiments)
    assert os.path.exists('sentiments.csv')