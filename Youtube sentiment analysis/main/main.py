import os
import googleapiclient.discovery
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

# Set up the YouTube API client
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = ""

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

def get_video_comments(video_id):
    comments = []
    results = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    ).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        
        if 'nextPageToken' in results:
            results = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,
                pageToken=results['nextPageToken']
            ).execute()
        else:
            break

    return comments

# Fetch comments
video_id = "d0o89z134CQ"
comments = get_video_comments(video_id)

# Create a DataFrame
df = pd.DataFrame({'comment': comments})

# Perform sentiment analysis
sia = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['comment'].apply(lambda x: sia.polarity_scores(x))
df['sentiment'] = df['sentiment_scores'].apply(lambda x: 'Positive' if x['compound'] > 0 else ('Negative' if x['compound'] < 0 else 'Neutral'))

# Display the first few rows of the DataFrame
print(df.head())

# Calculate sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
print("\
Sentiment Distribution:")
print(sentiment_counts)

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Distribution of YouTube Comments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('sentiment_distribution.png')
plt.close()

# Calculate average sentiment score
avg_sentiment = df['sentiment_scores'].apply(lambda x: x['compound']).mean()
print(f"\
Average Sentiment Score: {avg_sentiment:.2f}")

# Save results to CSV
df.to_csv('youtube_comments_sentiment.csv', index=False)
print("\
Results saved to 'youtube_comments_sentiment.csv'")

print("Sentiment distribution chart saved as 'sentiment_distribution.png'")
