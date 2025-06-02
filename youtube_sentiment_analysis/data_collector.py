import pandas as pd
from youtube_api import get_video_details, get_video_comments
from datetime import datetime
import csv

def collect_historical_data(video_ids):
    historical_data = []

    for video_id in video_ids:
        video_details = get_video_details(video_id)
        comments = get_video_comments(video_id, max_comments=500)  # Adjust as needed

        # Calculate engagement metrics
        likes = video_details['likes']
        views = video_details['views']
        sentiment_score = calculate_average_sentiment(comments)

        # Append data to the historical data list
        historical_data.append({
            'video_id': video_id,
            'title': video_details['title'],
            'description': video_details['description'],
            'upload_date': video_details['upload_date'],
            'likes': likes,
            'comments': len(comments),
            'views': views,
            'sentiment_score': sentiment_score,
            'popularity': views  # This can be adjusted based on your definition of popularity
        })

    # Convert to DataFrame
    df = pd.DataFrame(historical_data)

    # Save to CSV
    df.to_csv('historical_video_data.csv', index=False)
    print("Historical data collected and saved to historical_video_data.csv")

def calculate_average_sentiment(comments):
    # Placeholder for sentiment analysis logic
    # You can use your existing sentiment analysis functions here
    total_sentiment = sum(analyze_sentiment(comment['text']) for comment in comments)
    return total_sentiment / len(comments) if comments else 0

if __name__ == "__main__":
    video_ids = ['VIDEO_ID_1', 'VIDEO_ID_2', 'VIDEO_ID_3']  # Replace with actual video IDs
    collect_historical_data(video_ids)