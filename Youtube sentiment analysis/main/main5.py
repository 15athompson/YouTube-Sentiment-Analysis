import os
import googleapiclient.discovery
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud
import argparse
import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")
    sys.exit(1)

# Set up the YouTube API client
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = ""

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

def get_video_comments(video_id, max_comments=500):
    comments = []
    try:
        results = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        ).execute()

        while results and len(comments) < max_comments:
            for item in results['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
                if len(comments) >= max_comments:
                    break
            
            if 'nextPageToken' in results and len(comments) < max_comments:
                results = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=100,
                    pageToken=results['nextPageToken']
                ).execute()
            else:
                break
    except googleapiclient.errors.HttpError as e:
        logging.error(f"An error occurred while fetching comments for video {video_id}: {e}")
        return None

    return comments

def preprocess_text(text):
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text.lower())
        return [lemmatizer.lemmatize(word) for word in word_tokens if word.isalnum() and word not in stop_words]
    except LookupError:
        logging.warning("NLTK data not found. Falling back to basic preprocessing.")
        return [word.lower() for word in text.split() if word.isalnum()]

def analyze_sentiment(comment):
    try:
        sia = SentimentIntensityAnalyzer()
        blob = TextBlob(comment)
        
        sentiment_scores = sia.polarity_scores(comment)
        sentiment = 'Positive' if sentiment_scores['compound'] > 0 else ('Negative' if sentiment_scores['compound'] < 0 else 'Neutral')
        
        preprocessed_words = preprocess_text(comment)
        
        return {
            'comment': comment,
            'sentiment_scores': sentiment_scores,
            'sentiment': sentiment,
            'subjectivity': blob.sentiment.subjectivity,
            'word_count': len(blob.words),
            'preprocessed_words': preprocessed_words
        }
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return None

def process_video(video_id, max_comments):
    logging.info(f"Fetching comments for video ID: {video_id}")
    comments = get_video_comments(video_id, max_comments)
    
    if not comments:
        logging.warning(f"No comments found for video ID: {video_id}")
        return None
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(analyze_sentiment, comments), total=len(comments), desc="Analyzing comments"))
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    df = pd.DataFrame(results)
    return df

def visualize_sentiment(df, video_id):
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title(f'Sentiment Distribution of YouTube Comments (Video ID: {video_id})')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(f'output/{video_id}/sentiment_distribution.png')
    plt.close()

def generate_wordcloud(df, video_id):
    all_words = ' '.join([' '.join(words) for words in df['preprocessed_words']])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Video ID: {video_id}')
    plt.savefig(f'output/{video_id}/wordcloud.png')
    plt.close()

def analyze_top_words(df, video_id, top_n=10):
    all_words = [word for words in df['preprocessed_words'] for word in words]
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[word[1] for word in top_words], y=[word[0] for word in top_words])
    plt.title(f'Top {top_n} Words (Video ID: {video_id})')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.savefig(f'output/{video_id}/top_words.png')
    plt.close()
    
    return top_words

def perform_topic_modeling(df, video_id, num_topics=5):
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf = vectorizer.fit_transform(df['comment'])
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf)
    
    feature_names = vectorizer.get_feature_names_out()
    
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: " + ", ".join(top_words))
    
    return topics

def main(video_ids, max_comments):
    start_time = time.time()
    all_results = {}
    for video_id in video_ids:
        os.makedirs(f'output/{video_id}', exist_ok=True)
        df = process_video(video_id, max_comments)
        if df is not None:
            all_results[video_id] = df

            visualize_sentiment(df, video_id)
            generate_wordcloud(df, video_id)
            top_words = analyze_top_words(df, video_id)
            topics = perform_topic_modeling(df, video_id)
            
            avg_sentiment = df['sentiment_scores'].apply(lambda x: x['compound']).mean()
            avg_subjectivity = df['subjectivity'].mean()
            avg_word_count = df['word_count'].mean()

            results = {
                "video_id": video_id,
                "num_comments": len(df),
                "avg_sentiment": avg_sentiment,
                "avg_subjectivity": avg_subjectivity,
                "avg_word_count": avg_word_count,
                "top_words": dict(top_words),
                "topics": topics
            }

            with open(f'output/{video_id}/results.json', 'w') as f:
                json.dump(results, f, indent=4)

            logging.info(f"Analysis completed for Video ID: {video_id}")
            logging.info(f"Results saved to 'output/{video_id}/results.json'")
            logging.info(f"Visualizations saved in 'output/{video_id}/' directory")

    if len(all_results) > 1:
        compare_videos(all_results)

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

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
    plt.savefig('output/video_comparison.png')
    plt.close()

    logging.info("\nVideo Comparison:")
    logging.info(comparison_df)
    logging.info("Video comparison chart saved as 'output/video_comparison.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze YouTube video comments")
    parser.add_argument("video_ids", nargs="+", help="YouTube video IDs to analyze")
    parser.add_argument("--max_comments", type=int, default=500, help="Maximum number of comments to analyze per video")
    args = parser.parse_args()

    main(args.video_ids, args.max_comments)


