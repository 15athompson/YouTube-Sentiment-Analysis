import os
import json
import aiohttp
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

# ... (keep the existing imports)

# Add these new imports
import hashlib
from aiofiles import open as aopen
from aiofiles.os import makedirs

# ... (keep the existing code until the get_video_comments function)

CACHE_DIR = "comment_cache"
CACHE_EXPIRY_DAYS = 7

async def get_cached_comments(video_id: str) -> Optional[List[str]]:
    cache_file = os.path.join(CACHE_DIR, f"{video_id}.json")
    try:
        async with aopen(cache_file, "r") as f:
            cache_data = json.loads(await f.read())
        
        # Check if cache is expired
        cache_date = datetime.fromisoformat(cache_data['cache_date'])
        if datetime.now() - cache_date > timedelta(days=CACHE_EXPIRY_DAYS):
            return None
        
        return cache_data['comments']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

async def save_comments_to_cache(video_id: str, comments: List[str]) -> None:
    cache_file = os.path.join(CACHE_DIR, f"{video_id}.json")
    cache_data = {
        'cache_date': datetime.now().isoformat(),
        'comments': comments
    }
    await makedirs(CACHE_DIR, exist_ok=True)
    async with aopen(cache_file, "w") as f:
        await f.write(json.dumps(cache_data, indent=2))

async def get_video_comments(session: aiohttp.ClientSession, video_id: str, max_comments: int = 500) -> Optional[List[str]]:
    # Check cache first
    cached_comments = await get_cached_comments(video_id)
    if cached_comments is not None:
        logging.info(f"Using cached comments for video ID: {video_id}")
        return cached_comments[:max_comments]

    comments = []
    next_page_token = None
    url = f"https://www.googleapis.com/youtube/v3/commentThreads"
    
    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "textFormat": "plainText",
            "maxResults": min(100, max_comments - len(comments)),
            "key": DEVELOPER_KEY
        }
        if next_page_token:
            params["pageToken"] = next_page_token
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logging.error(f"Error fetching comments for video {video_id}: HTTP {response.status}")
                    return None
                data = await response.json()
                
                for item in data['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)
                
                next_page_token = data.get('nextPageToken')
                if not next_page_token or len(comments) >= max_comments:
                    break
        except Exception as e:
            logging.error(f"An error occurred while fetching comments for video {video_id}: {e}")
            return None
    
    # Save to cache
    await save_comments_to_cache(video_id, comments)
    
    return comments

# ... (keep the rest of the existing code)

async def main(video_ids: List[str], max_comments: int) -> None:
    start_time = time.time()
    all_results = {}
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_video(session, video_id, max_comments) for video_id in video_ids]
        results = await asyncio.gather(*tasks)
    
    for video_id, df in zip(video_ids, results):
        if df is not None:
            os.makedirs(f'output/{video_id}', exist_ok=True)
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

# ... (keep the rest of the existing code)