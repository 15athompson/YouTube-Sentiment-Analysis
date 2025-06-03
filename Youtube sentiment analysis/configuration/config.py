# YouTube API settings
youtube_api_key: YOUR_API_KEY_HERE

# Analysis settings
max_comments: 500
num_topics: 5
top_words: 10

# NLP settings
perform_ner: true
perform_advanced_sentiment: true
sentiment_model: "distilbert-base-uncased-finetuned-sst-2-english"

# Visualization settings
wordcloud_width: 800
wordcloud_height: 400
chart_width: 10
chart_height: 6

# Caching settings
cache_dir: "comment_cache"
cache_expiry_days: 7

# Output settings
output_dir: "output"

# Logging settings
log_level: INFO