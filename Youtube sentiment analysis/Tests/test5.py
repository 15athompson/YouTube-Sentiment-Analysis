# Import Necessary Libraries

import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

# Replace with your YouTube Data API credentials
api_key = "YOUR_API_KEY"

# Authenticate with YouTube Data API

def authenticate():
  creds = Credentials.from_authorized_user_file('client_secret.json')
  youtube = build('youtube', 'v3', credentials=creds)
  return youtube

youtube = authenticate()

# Fetch Video Data

def fetch_video_data(search_query):
  request = youtube.search().list(
      part="snippet",
      q=search_query,
      maxResults=50
  )
  response = request.execute()

  video_ids = [item['id']['videoId'] for item in response['items']]
  videos = []

  for video_id in video_ids:
      request = youtube.videos().list(
          part="snippet, statistics",
          id=video_id
      )
      response = request.execute()

      videos.append(response['items'][0])

  return videos

search_query = "gaming videos"
videos = fetch_video_data(search_query)

# Extract and Analyze Sentiment

def analyze_sentiment(text):
  blob = TextBlob(text)
  sentiment = blob.sentiment
  return sentiment.polarity, sentiment.subjectivity

video_data = []

for video in videos:
  title = video['snippet']['title']
  description = video['snippet']['description']
  view_count = video['statistics']['viewCount']
  like_count = video['statistics']['likeCount']
  dislike_count = video['statistics']['dislikeCount']

  title_polarity, title_subjectivity = analyze_sentiment(title)
  description_polarity, description_subjectivity = analyze_sentiment(description)

  video_data.append({
      'title': title,
      'description': description,
      'view_count': view_count,
      'like_count': like_count,
      'dislike_count': dislike_count,
      'title_polarity': title_polarity,
      'description_polarity': description_polarity,
      'title_subjectivity': title_subjectivity,
      'description_subjectivity': description_subjectivity
  })

df = pd.DataFrame(video_data)

# Visualization and Analysis

# Visualize sentiment distribution
sns.histplot(df['title_polarity'], bins=30, kde=True)
plt.title('Title Sentiment Distribution')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.show()

# Analyze correlation between sentiment and view count
correlation = df['title_polarity'].corr(df['view_count'])
print("Correlation between title polarity and view count:", correlation)