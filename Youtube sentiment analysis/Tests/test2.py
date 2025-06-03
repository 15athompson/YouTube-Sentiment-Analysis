import os
import json
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import csv

# Set up YouTube API credentials
api_key = os.environ['YOUTUBE_API_KEY']
youtube = build('youtube', 'v3', developerKey=api_key)

# Set up sentiment analysis
nlp = spacy.load('en_core_web_sm')

def fetch_comments(video_id):
    comments = []
    next_page_token = ''
    while True:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            pageToken=next_page_token
        )
        try:
            response = request.execute()
        except HttpError as e:
            print(f"Error: {e}")
            break
        for item in response['items']:
            comment = item['snippet']['topLevelComment']
            text = comment['snippet']['textDisplay']
            comments.append(text)
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return comments

def analyze_sentiment(comments):
    sentiments = []
    for comment in comments:
        doc = nlp(comment)
        sentiment = doc._.polarity
        sentiments.append(sentiment)
    return sentiments

def store_results(sentiments):
    with open('sentiments.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['polarity', 'subjectivity'])
        for sentiment in sentiments:
            writer.writerow([sentiment['polarity'], sentiment['subjectivity']])

video_id = input("Enter the video ID: ")
comments = fetch_comments(video_id)
sentiments = analyze_sentiment(comments)
store_results(sentiments)