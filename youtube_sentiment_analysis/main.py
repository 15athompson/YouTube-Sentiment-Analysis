import asyncio
import argparse
import logging
import json
from typing import List, Dict
from datetime import datetime, timedelta
from youtube_api import get_video_comments, get_video_details, get_playlist_videos, get_comment_replies
from sentiment_analyzer import analyze_sentiment, analyze_emojis, analyze_text_sentiment, analyze_sentiment_trend, deep_analyze_comment, analyze_topic_sentiment, analyze_keyword_sentiment
from topic_modeling import perform_lda_topic_modeling
from data_visualizer import (
    visualize_sentiment, create_word_cloud, visualize_sentiment_over_time,
    save_visualizations, visualize_sentiment_trend, compare_time_periods,
    visualize_topic_distribution, visualize_named_entities, visualize_sentiment_aspects,
    visualize_transcript_sentiment, visualize_audio_sentiment, visualize_sentiment_comparison
)
from data_saver import save_to_csv, save_to_json, save_to_database
from cache_manager import CacheManager
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from fastapi import FastAPI, BackgroundTasks
import uvicorn
from aiohttp import ClientSession
from tenacity import retry, stop_after_attempt, wait_exponential
from .sentiment_analysis import analyze_comments
from transcript_audio_analysis import analyze_transcript_and_audio
from popularity_predictor import PopularityPredictor
import pandas as pd
from twitter_api import fetch_tweets
from reddit_api import fetch_reddit_posts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_with_retry(session, url, **kwargs):
    async with session.get(url, **kwargs) as response:
        response.raise_for_status()
        return await response.json()

async def process_video(video_id: str, max_comments: int, cache_manager: CacheManager, start_date: str = None, end_date: str = None, min_likes: int = 0, include_replies: bool = False, topics: List[str] = None, keywords: List[str] = None) -> Dict:
    try:
        cached_result = cache_manager.get_results(video_id)
        if cached_result and not cache_manager.is_expired(video_id):
            logger.info(f"Using cached results for video {video_id}")
            return cached_result

        async with ClientSession() as session:
            video_details = await get_video_details(session, video_id)
            comments = await get_video_comments(session, video_id, max_comments, start_date, end_date, min_likes)
            
            if include_replies:
                for comment in comments:
                    replies = await get_comment_replies(session, comment['id'])
                    comment['replies'] = replies

        if not comments:
            logger.warning(f"No comments found or error occurred for video {video_id}.")
            return None

        with ThreadPoolExecutor() as executor:
            sentiment_results = list(tqdm(
                executor.map(analyze_sentiment, comments),
                total=len(comments),
                desc=f"Analyzing sentiment for video {video_id}"
            ))

        emoji_analysis = analyze_emojis(comments)
        title_sentiment = analyze_text_sentiment(video_details['title'])
        description_sentiment = analyze_text_sentiment(video_details['description'])
        sentiment_trend = analyze_sentiment_trend(sentiment_results)

        # Perform LDA topic modeling
        topics = perform_lda_topic_modeling([comment['text'] for comment in comments])

        if keywords:
            keyword_sentiment = analyze_keyword_sentiment(comments, keywords)
        else:
            keyword_sentiment = None

        analysis_results = analyze_comments(comments)
        
        # Perform transcript and audio analysis
        transcript_audio_results = analyze_transcript_and_audio(video_id)
        
        # Fetch tweets and Reddit posts
        tweets = fetch_tweets(video_details['title'], count=max_comments)
        reddit_posts = fetch_reddit_posts('your_subreddit', limit=max_comments)

        # Analyze sentiment from social media
        social_media_sentiment = analyze_social_media_sentiment(tweets, reddit_posts)

        # Combine with existing sentiment results
        sentiment_results = sentiment_results + social_media_sentiment
        
        # After processing the video and gathering results
        popularity_predictor = PopularityPredictor()
        
        # Prepare data for training or prediction
        # This is a placeholder; you need to gather historical data
        historical_data = pd.DataFrame({
            'likes': [100, 200, 300],  # Example features
            'comments': [10, 20, 30],
            'sentiment_score': [0.5, 0.6, 0.7],
            'popularity': [1000, 2000, 3000]  # Target variable
        })

        # Train the model (you might want to do this separately)
        popularity_predictor.train(historical_data)

        # Predict popularity for the current video
        features = [video_details['likes'], len(comments), sentiment_results[0]['compound']]  # Example features
        predicted_popularity = popularity_predictor.predict(features)

        result = {
            'video_id': video_id,
            'title': video_details['title'],
            'sentiment_results': sentiment_results,
            'emoji_analysis': emoji_analysis,
            'title_sentiment': title_sentiment,
            'description_sentiment': description_sentiment,
            'sentiment_trend': sentiment_trend,
            'topics': topics,
            'keyword_sentiment': keyword_sentiment,
            'named_entities': analysis_results['advanced_analysis']['named_entities'],
            'sentiment_aspects': analysis_results['advanced_analysis']['sentiment_aspects'],
            'transcript': transcript_audio_results['transcript'],
            'transcript_sentiment': transcript_audio_results['transcript_sentiment'],
            'audio_transcript': transcript_audio_results['audio_transcript'],
            'audio_sentiment': transcript_audio_results['audio_sentiment'],
            'predicted_popularity': predicted_popularity[0]  # Add prediction to result
        }

        cache_manager.save_results(video_id, result)
        return result
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        return None

async def process_videos(video_ids: List[str], max_comments: int, cache_manager: CacheManager, start_date: str = None, end_date: str = None, min_likes: int = 0, include_replies: bool = False, keywords: List[str] = None) -> List[Dict]:
    tasks = [process_video(video_id, max_comments, cache_manager, start_date, end_date, min_likes, include_replies, keywords) for video_id in video_ids]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

async def analyze_sentiment_trends(results: List[Dict]):
    """
    Analyze sentiment trends across multiple videos.
    This function will aggregate sentiment scores and calculate averages.
    """
    # Initialize a dictionary to hold aggregated sentiment data
    aggregated_data = {}

    for result in results:
        # Extract the date and compound sentiment scores
        for sentiment in result['sentiment_results']:
            date = sentiment['date'][:10]  # Get the date part (YYYY-MM-DD)
            compound_score = sentiment['compound']

            if date not in aggregated_data:
                aggregated_data[date] = {
                    'total_score': 0,
                    'count': 0
                }

            # Aggregate the scores
            aggregated_data[date]['total_score'] += compound_score
            aggregated_data[date]['count'] += 1

    # Calculate average scores
    average_trend = {
        'dates': [],
        'average_scores': []
    }

    for date, data in sorted(aggregated_data.items()):
        average_score = data['total_score'] / data['count']
        average_trend['dates'].append(date)
        average_trend['average_scores'].append(average_score)

    # You can now visualize this average_trend data using your existing visualization functions
    return average_trend

async def main():
    parser = argparse.ArgumentParser(description="YouTube Comment Sentiment Analyzer")
    parser.add_argument("input", nargs='+', help="YouTube video ID(s) or playlist ID(s)")
    parser.add_argument("--playlist", action="store_true", help="Analyze playlist(s) instead of individual videos")
    parser.add_argument("--max_comments", type=int, default=500, help="Maximum number of comments to analyze per video")
    parser.add_argument("--output", default=None, help="Output file name (without extension)")
    parser.add_argument("--start_date", help="Start date for comment filtering (YYYY-MM-DD)")
    parser.add_argument("--end_date", help="End date for comment filtering (YYYY-MM-DD)")
    parser.add_argument("--min_likes", type=int, default=0, help="Minimum number of likes for comments to be included")
    parser.add_argument("--deep_analysis", type=int, default=0, help="Number of top comments to perform deep analysis on")
    parser.add_argument("--include_replies", action="store_true", help="Include comment replies in the analysis")
    parser.add_argument("--keywords", nargs='+', help="Specific keywords to analyze sentiment for")
    parser.add_argument("--compare_periods", action="store_true", help="Compare sentiment between two time periods")
    parser.add_argument("--database", help="SQLite database file to save results")
    args = parser.parse_args()

    cache_manager = CacheManager()

    all_video_ids = []
    async with ClientSession() as session:
        for input_id in args.input:
            if args.playlist:
                video_ids = await get_playlist_videos(session, input_id)
                all_video_ids.extend(video_ids)
            else:
                all_video_ids.append(input_id)

    if args.compare_periods:
        if len(all_video_ids) != 1:
            logger.error("Please provide exactly one video ID when comparing time periods.")
            return
        
        video_id = all_video_ids[0]
        period1 = {
            'start_date': args.start_date,
            'end_date': (datetime.strptime(args.start_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d"),
            'max_comments': args.max_comments,
            'min_likes': args.min_likes
        }
        period2 = {
            'start_date': args.end_date,
            'end_date': (datetime.strptime(args.end_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d"),
            'max_comments': args.max_comments,
            'min_likes': args.min_likes
        }
        await compare_time_periods(video_id, period1, period2, cache_manager)
    else:
        results = await process_videos(all_video_ids, args.max_comments, cache_manager, args.start_date, args.end_date, args.min_likes, args.include_replies, args.keywords)

        if not results:
            logger.error("No results to analyze.")
            return

        # Analyze sentiment trends across all results
        sentiment_trend = await analyze_sentiment_trends(results)

        for result in results:
            # Add the sentiment trend to each result
            result['sentiment_trend'] = sentiment_trend

            logger.info(f"\nAnalyzing video: {result['title']} ({result['video_id']})")
            logger.info(f"Title sentiment: {result['title_sentiment']}")
            logger.info(f"Description sentiment: {result['description_sentiment']}")
            logger.info(f"Transcript sentiment: {result['transcript_sentiment']}")
            logger.info(f"Audio sentiment: {result['audio_sentiment']}")
            
            visualize_sentiment(result['sentiment_results'])
            create_word_cloud([comment['text'] for comment in result['sentiment_results']])
            visualize_sentiment_over_time(result['sentiment_results'])
            visualize_sentiment_trend(result['sentiment_trend'])
            visualize_topic_distribution(result['topics'])
            visualize_named_entities(result['named_entities'])
            visualize_sentiment_aspects(result['sentiment_aspects'])
            
            # New visualizations
            visualize_transcript_sentiment(result['transcript_sentiment'])
            visualize_audio_sentiment(result['audio_sentiment'])
            
            average_comment_sentiment = sum(r['compound'] for r in result['sentiment_results']) / len(result['sentiment_results'])
            visualize_sentiment_comparison(average_comment_sentiment, result['transcript_sentiment'], result['audio_sentiment'])

            if args.keywords:
                for keyword, sentiment in result['keyword_sentiment'].items():
                    logger.info(f"Sentiment for keyword '{keyword}': {sentiment}")

            if args.deep_analysis > 0:
                top_comments = sorted(result['sentiment_results'], key=lambda x: x['likes'], reverse=True)[:args.deep_analysis]
                for comment in top_comments:
                    deep_analysis = deep_analyze_comment(comment['comment'])
                    logger.info(f"Deep analysis for comment: {comment['comment'][:50]}...")
                    logger.info(json.dumps(deep_analysis, indent=2))

        if args.output:
            save_to_csv(f"{args.output}.csv", results)
            save_to_json(f"{args.output}.json", results)
            save_visualizations(results)
            logger.info(f"Results saved to {args.output}.csv, {args.output}.json, and visualizations folder")

        if args.database:
            save_to_database(args.database, results)
            logger.info(f"Results saved to database: {args.database}")

@app.post("/analyze")
async def analyze_video(video_id: str, background_tasks: BackgroundTasks):
    cache_manager = CacheManager()
    background_tasks.add_task(process_video, video_id, 500, cache_manager)
    return {"message": "Analysis started", "video_id": video_id}

@app.get("/results/{video_id}")
async def get_results(video_id: str):
    cache_manager = CacheManager()
    results = cache_manager.get_results(video_id)
    if results:
        # Ensure all necessary data is included
        response = {
            "video_id": results["video_id"],
            "title": results["title"],
            "sentiment_results": results["sentiment_results"],
            "topics": results["topics"],
            "named_entities": results["named_entities"],
            "sentiment_aspects": results["sentiment_aspects"],
            "transcript_sentiment": results["transcript_sentiment"],
            "audio_sentiment": results["audio_sentiment"],
            "sentiment_trend": results["sentiment_trend"]
        }
        return response
    else:
        return {"message": "Results not found or still processing"}

async def analyze_video_for_dashboard(video_id: str, max_comments: int, cache_manager: CacheManager) -> Dict:
    try:
        result = await process_video(video_id, max_comments, cache_manager)
        return result
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        return None

if __name__ == "__main__":
    asyncio.run(main())

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        run_api()
    else:
        asyncio.run(main())