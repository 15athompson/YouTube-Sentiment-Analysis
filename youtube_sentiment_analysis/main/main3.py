import os
import googleapiclient.discovery
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from textblob import TextBlob

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Set up the YouTube API client
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyA7G0j1IqNbGMbsfB-JChhnd7hOH1exRG0"

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

def analyze_sentiment(comment):
    sia = SentimentIntensityAnalyzer()
    blob = TextBlob(comment)
    
    sentiment_scores = sia.polarity_scores(comment)
    sentiment = 'Positive' if sentiment_scores['compound'] > 0 else ('Negative' if sentiment_scores['compound'] < 0 else 'Neutral')
    
    return {
        'comment': comment,
        'sentiment_scores': sentiment_scores,
        'sentiment': sentiment,
        'subjectivity': blob.sentiment.subjectivity,
        'word_count': len(blob.words)
    }

def process_video(video_id):
    print(f"Fetching comments for video ID: {video_id}")
    comments = get_video_comments(video_id)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(analyze_sentiment, comments), total=len(comments), desc="Analyzing comments"))
    
    df = pd.DataFrame(results)
    return df

def visualize_sentiment(df, video_id):
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title(f'Sentiment Distribution of YouTube Comments (Video ID: {video_id})')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(f'sentiment_distribution_{video_id}.png')
    plt.close()

def main():
    video_ids = input("Enter YouTube video IDs (comma-separated): ").split(',')
    video_ids = [vid.strip() for vid in video_ids]

    all_results = {}
    for video_id in video_ids:
        df = process_video(video_id)
        all_results[video_id] = df

        visualize_sentiment(df, video_id)
        
        avg_sentiment = df['sentiment_scores'].apply(lambda x: x['compound']).mean()
        avg_subjectivity = df['subjectivity'].mean()
        avg_word_count = df['word_count'].mean()

        print(f"\nResults for Video ID: {video_id}")
        print(f"Number of comments analyzed: {len(df)}")
        print(f"Average Sentiment Score: {avg_sentiment:.2f}")
        print(f"Average Subjectivity Score: {avg_subjectivity:.2f}")
        print(f"Average Word Count: {avg_word_count:.2f}")
        
        df.to_csv(f'youtube_comments_sentiment_{video_id}.csv', index=False)
        print(f"Results saved to 'youtube_comments_sentiment_{video_id}.csv'")
        print(f"Sentiment distribution chart saved as 'sentiment_distribution_{video_id}.png'")

    if len(video_ids) > 1:
        compare_videos(all_results)

def compare_videos(all_results):
    comparison_data = []
    for video_id, df in all_results.items():
        avg_sentiment = df['sentiment_scores'].apply(lambda x: x['compound']).mean()
        avg_subjectivity = df['subjectivity'].mean()
        comparison_data.append({
            'video_id': video_id,
            'avg_sentiment': avg_sentiment,
            'avg_subjectivity': avg_subjectivity
        })

    comparison_df = pd.DataFrame(comparison_data)
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=comparison_df, x='avg_sentiment', y='avg_subjectivity', s=100)
    for i, row in comparison_df.iterrows():
        plt.annotate(row['video_id'], (row['avg_sentiment'], row['avg_subjectivity']))
    plt.title('Video Comparison: Average Sentiment vs Subjectivity')
    plt.xlabel('Average Sentiment')
    plt.ylabel('Average Subjectivity')
    plt.savefig('video_comparison.png')
    plt.close()

    print("\nVideo Comparison:")
    print(comparison_df)
    print("Video comparison chart saved as 'video_comparison.png'")

if __name__ == "__main__":
    main()