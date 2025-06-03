import os
import googleapiclient.discovery
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures

nltk.download('vader_lexicon', quiet=True)

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyA7G0j1IqNbGMbsfB-JChhnd7hOH1exRG0"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

class VideoCommentAnalyzer:
    def __init__(self, video_ids):
        self.video_ids = video_ids
        self._comments_dataframes = None

    @property
    def comments_dataframes(self):
        if not self._comments_dataframes:
            self._fetch_and_analyze_comments()
        return self._comments_dataframes

    def _fetch_and_analyze_comments(self):
        self._comments_dataframes = []
        executor = concurrent.futures.ThreadPoolExecutor()

        futures = {executor.submit(self._get_video_comments, vid): vid for vid in self.video_ids}
        for future in concurrent.futures.as_completed(futures):
            video_id = futures[future]
            df = pd.DataFrame(future.result(), columns=['comment'])
            df['video_id'] = video_id
            self._analyze_sentiments(df)
            self._comments_dataframes.append(df)

    def _get_video_comments(self, video_id):
        comments = []
        results = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        ).execute()


    def _analyze_sentiments(self, df):
        sia = SentimentIntensityAnalyzer()
        df['sentiment_scores'] = df['comment'].apply(lambda x: sia.polarity_scores(x))
        df['sentiment'] = df['sentiment_scores'].apply(lambda x: 'Positive' if x['compound'] > 0 else ('Negative' if x['compound'] < 0 else 'Neutral'))

if __name__ == "__main__":
    video_ids = ['d0o89z134CQ', 'tq9yN-j5o3M']  # add more video ids here
    analyzer = VideoCommentAnalyzer(video_ids)
    combined_df = pd.concat(analyzer.comments_dataframes)
    
    print(combined_df.head())

    sentiment_counts = combined_df['sentiment'].value_counts()
    print("\nSentiment Distribution:\n", sentiment_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Distribution of Combined YouTube Comments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    avg_sentiment = combined_df['sentiment_scores'].apply(lambda x: x['compound']).mean()
    print(f"\nAverage Sentiment Score: {avg_sentiment:.2f}\n")

    combined_df.to_csv('combined_youtube_comments_sentiment.csv', index=False)
    print("Results saved to 'combined_youtube_comments_sentiment.csv'\n")