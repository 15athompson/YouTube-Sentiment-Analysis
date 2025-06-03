import googleapiclient.discovery
from textblob import TextBlob

# Set up YouTube API client
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "YOUR_API_KEY"
youtube = googleapiclient.discovery.build(
  api_service_name, api_version, developerKey = DEVELOPER_KEY)

# Define function to get video comments
def get_video_comments(video_id):
  comments = []
  results = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    textFormat="plainText"
  ).execute()
  for item in results["items"]:
    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
    comments.append(comment)
  return comments

# Define function to perform sentiment analysis on comments
def analyze_sentiment(comments):
  sentiment_scores = []
  for comment in comments:
    blob = TextBlob(comment)
    sentiment_scores.append(blob.sentiment.polarity)
  return sentiment_scores

# Example usage
video_id = "YOUR_VIDEO_ID"
comments = get_video_comments(video_id)
sentiment_scores = analyze_sentiment(comments)
print(sentiment_scores)