import os
import json
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set up YouTube API credentials
api_key = 'YOUR_API_KEY'
youtube = build('youtube', 'v3', developerKey=api_key)

# Set up sentiment analysis
sia = SentimentIntensityAnalyzer()

# Fetch comments from a video
video_id = 'VIDEO_ID'
comments = []
next_page_token = ''
while True:
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        pageToken=next_page_token
    )
    response = request.execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']
        text = comment['snippet']['textDisplay']
        comments.append(text)
    next_page_token = response.get('nextPageToken')
    if not next_page_token:
        break

# Analyze sentiment
sentiments = []
for comment in comments:
    sentiment = sia.polarity_scores(comment)
    sentiments.append(sentiment)

# Store and visualize the results
df = pd.DataFrame(sentiments)
print(df.head())