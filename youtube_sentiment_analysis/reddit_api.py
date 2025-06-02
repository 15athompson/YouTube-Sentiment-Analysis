import praw

def fetch_reddit_posts(subreddit, limit=100):
    # Reddit API credentials
    reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                         client_secret='YOUR_CLIENT_SECRET',
                         user_agent='YOUR_USER_AGENT')

    # Fetch posts
    posts = reddit.subreddit(subreddit).new(limit=limit)
    return [{'text': post.title + ' ' + post.selftext, 'date': post.created_utc} for post in posts]