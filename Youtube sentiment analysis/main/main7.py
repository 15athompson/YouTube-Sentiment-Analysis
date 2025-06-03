import os
import yaml
import googleapiclient.discovery
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
from wordcloud import WordCloud
import argparse
import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging
import time
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from aiofiles import open as aopen
from aiofiles.os import makedirs

# New imports for advanced NLP
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load configuration
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Set up logging
logging.basicConfig(level=config['log_level'], format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")
    sys.exit(1)

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained(config['sentiment_model'])
model = AutoModelForSequenceClassification.from_pretrained(config['sentiment_model'])
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# ... (keep the YouTube API setup and caching functions)

def preprocess_text(text: str) -> List[str]:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(word) for word in word_tokens if word.isalnum() and word not in stop_words]

def perform_ner(text: str) -> List[Dict[str, str]]:
    doc = nlp(text)
    return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

def analyze_sentiment_advanced(text: str) -> Dict[str, Any]:
    result = sentiment_pipeline(text)[0]
    return {
        'label': result['label'],
        'score': result['score']
    }

def analyze_comment(comment: str) -> Dict[str, Any]:
    preprocessed_words = preprocess_text(comment)
    
    analysis = {
        'comment': comment,
        'preprocessed_words': preprocessed_words,
        'word_count': len(preprocessed_words)
    }
    
    if config['perform_ner']:
        analysis['named_entities'] = perform_ner(comment)
    
    if config['perform_advanced_sentiment']:
        analysis['sentiment'] = analyze_sentiment_advanced(comment)
    
    return analysis

async def process_video(session: aiohttp.ClientSession, video_id: str) -> Optional[pd.DataFrame]:
    logging.info(f"Fetching comments for video ID: {video_id}")
    comments = await get_video_comments(session, video_id)
    
    if not comments:
        logging.warning(f"No comments found for video ID: {video_id}")
        return None
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(analyze_comment, comments), total=len(comments), desc="Analyzing comments"))
    
    df = pd.DataFrame(results)
    return df

def visualize_sentiment(df: pd.DataFrame, video_id: str) -> None:
    sentiment_counts = df['sentiment'].apply(lambda x: x['label']).value_counts()
    plt.figure(figsize=(config['chart_width'], config['chart_height']))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title(f'Sentiment Distribution of YouTube Comments (Video ID: {video_id})')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(f'{config["output_dir"]}/{video_id}/sentiment_distribution.png')
    plt.close()

def visualize_named_entities(df: pd.DataFrame, video_id: str) -> None:
    all_entities = [entity for entities in df['named_entities'] for entity in entities]
    entity_counts = Counter([entity['label'] for entity in all_entities])
    
    plt.figure(figsize=(config['chart_width'], config['chart_height']))
    sns.barplot(x=list(entity_counts.keys()), y=list(entity_counts.values()))
    plt.title(f'Named Entity Distribution (Video ID: {video_id})')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{config["output_dir"]}/{video_id}/named_entity_distribution.png')
    plt.close()

# ... (keep other visualization functions like generate_wordcloud, analyze_top_words, etc.)

async def main(video_ids: List[str]) -> None:
    start_time = time.time()
    all_results = {}
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_video(session, video_id) for video_id in video_ids]
        results = await asyncio.gather(*tasks)
    
    for video_id, df in zip(video_ids, results):
        if df is not None:
            os.makedirs(f'{config["output_dir"]}/{video_id}', exist_ok=True)
            all_results[video_id] = df

            visualize_sentiment(df, video_id)
            if config['perform_ner']:
                visualize_named_entities(df, video_id)
            generate_wordcloud(df, video_id)
            top_words = analyze_top_words(df, video_id)
            topics = perform_topic_modeling(df, video_id)
            
            avg_sentiment = df['sentiment'].apply(lambda x: x['score']).mean()
            avg_word_count = df['word_count'].mean()

            results = {
                "video_id": video_id,
                "num_comments": len(df),
                "avg_sentiment_score": avg_sentiment,
                "avg_word_count": avg_word_count,
                "top_words": dict(top_words),
                "topics": topics
            }

            with open(f'{config["output_dir"]}/{video_id}/results.json', 'w') as f:
                json.dump(results, f, indent=4)

            logging.info(f"Analysis completed for Video ID: {video_id}")
            logging.info(f"Results saved to '{config['output_dir']}/{video_id}/results.json'")
            logging.info(f"Visualizations saved in '{config['output_dir']}/{video_id}/' directory")

    if len(all_results) > 1:
        compare_videos(all_results)

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze YouTube video comments")
    parser.add_argument("video_ids", nargs="+", help="YouTube video IDs to analyze")
    args = parser.parse_args()

    asyncio.run(main(args.video_ids))