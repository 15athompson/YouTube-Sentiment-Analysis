import os
import asyncio
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm
from datetime import datetime

API_KEY = os.environ.get("YOUTUBE_API_KEY")

async def get_video_details(video_id):
    if not API_KEY:
        raise ValueError("YouTube API key not found. Set the YOUTUBE_API_KEY environment variable.")

    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    try:
        response = await asyncio.to_thread(
            youtube.videos().list(
                part="snippet",
                id=video_id
            ).execute
        )

        if response['items']:
            video_data = response['items'][0]['snippet']
            return {
                'title': video_data['title'],
                'description': video_data['description']
            }
        else:
            print(f"No video details found for video ID: {video_id}")
            return None
    except HttpError as e:
        print(f"An error occurred while fetching video details: {e}")
        return None

async def get_video_comments(video_id, max_results=500, start_date=None, end_date=None, min_likes=0):
    if not API_KEY:
        raise ValueError("YouTube API key not found. Set the YOUTUBE_API_KEY environment variable.")

    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    comments = []
    next_page_token = None

    if start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    try:
        with tqdm(total=max_results, desc="Fetching comments") as pbar:
            while len(comments) < max_results:
                response = await asyncio.to_thread(
                    youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=min(100, max_results - len(comments)),
                        pageToken=next_page_token
                    ).execute
                )

                for item in response["items"]:
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    comment_date = datetime.strptime(comment["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
                    
                    if (not start_date or comment_date >= start_date) and \
                       (not end_date or comment_date <= end_date) and \
                       comment["likeCount"] >= min_likes:
                        comments.append({
                            'text': comment["textDisplay"],
                            'date': comment["publishedAt"],
                            'likes': comment["likeCount"]
                        })
                        pbar.update(1)

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

        return comments
    except HttpError as e:
        print(f"An error occurred: {e}")
        return []

async def get_playlist_videos(playlist_id):
    if not API_KEY:
        raise ValueError("YouTube API key not found. Set the YOUTUBE_API_KEY environment variable.")

    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    video_ids = []
    next_page_token = None

    try:
        while True:
            response = await asyncio.to_thread(
                youtube.playlistItems().list(
                    part="snippet",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                ).execute
            )

            for item in response["items"]:
                video_ids.append(item["snippet"]["resourceId"]["videoId"])

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return video_ids
    except HttpError as e:
        print(f"An error occurred while fetching playlist videos: {e}")
        return []

# Add this function to the existing youtube_api.py file

async def get_comment_replies(session, comment_id, max_results=100):
    if not API_KEY:
        raise ValueError("YouTube API key not found. Set the YOUTUBE_API_KEY environment variable.")

    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    replies = []
    next_page_token = None

    try:
        while len(replies) < max_results:
            response = await asyncio.to_thread(
                youtube.comments().list(
                    part="snippet",
                    parentId=comment_id,
                    maxResults=min(100, max_results - len(replies)),
                    pageToken=next_page_token
                ).execute
            )

            for item in response["items"]:
                reply = item["snippet"]
                replies.append({
                    'text': reply["textDisplay"],
                    'date': reply["publishedAt"],
                    'likes': reply["likeCount"]
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return replies
    except HttpError as e:
        print(f"An error occurred while fetching comment replies: {e}")
        return []