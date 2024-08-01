from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os

class YouTubeDataExtractor:
    def __init__(self):
        api_key = os.environ.get('YOUTUBE_API_KEY')
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY environment variable is not set")
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    def extract_comments(self, video_id, max_comments=500):
        comments = []
        next_page_token = None
        total_comments = 0

        while total_comments < max_comments:
            try:
                results = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=min(100, max_comments - total_comments),
                    pageToken=next_page_token
                ).execute()

                for item in results["items"]:
                    comment_snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comment = comment_snippet["textDisplay"]
                    comment_id = item["id"]
                    publish_date = comment_snippet["publishedAt"]
                    replies = self.fetch_comment_replies(comment_id, max_comments - total_comments)
                    
                    comments.append({
                        "text": comment,
                        "publish_date": publish_date,
                        "replies": replies
                    })
                    
                    total_comments += 1 + len(replies)
                    if total_comments >= max_comments:
                        break

                next_page_token = results.get("nextPageToken")
                if not next_page_token:
                    break  # No more comments to fetch

            except HttpError as e:
                print(f"An HTTP error {e.resp.status} occurred: {e.content}")
                break  # Stop fetching comments if an error occurs

        print(f"Fetched {total_comments} comments and replies for video {video_id}")
        return comments

    def fetch_comment_replies(self, parent_id, max_replies):
        replies = []
        next_page_token = None

        while len(replies) < max_replies:
            try:
                results = self.youtube.comments().list(
                    part="snippet",
                    parentId=parent_id,
                    textFormat="plainText",
                    maxResults=min(100, max_replies - len(replies)),
                    pageToken=next_page_token
                ).execute()

                for item in results["items"]:
                    reply_snippet = item["snippet"]
                    reply = reply_snippet["textDisplay"]
                    publish_date = reply_snippet["publishedAt"]
                    replies.append({
                        "text": reply,
                        "publish_date": publish_date
                    })

                next_page_token = results.get("nextPageToken")
                if not next_page_token:
                    break  # No more replies to fetch

            except HttpError as e:
                print(f"An HTTP error {e.resp.status} occurred while fetching replies: {e.content}")
                break  # Stop fetching replies if an error occurs

        return replies

    def get_channel_videos(self, channel_id, max_videos=50):
        videos = []
        next_page_token = None

        try:
            # First, get the channel's uploads playlist ID
            channel_response = self.youtube.channels().list(
                part="contentDetails",
                id=channel_id
            ).execute()

            if not channel_response['items']:
                raise ValueError(f"No channel found with ID: {channel_id}")

            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            # Now, get the videos from this playlist
            while len(videos) < max_videos:
                playlist_response = self.youtube.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_videos - len(videos)),
                    pageToken=next_page_token
                ).execute()

                for item in playlist_response['items']:
                    video_id = item['snippet']['resourceId']['videoId']
                    publish_date = item['snippet']['publishedAt']
                    videos.append((video_id, publish_date))

                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break

        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred: {e.content}")

        print(f"Fetched {len(videos)} videos from channel {channel_id}")
        return videos
