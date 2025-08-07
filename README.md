# YouTube Sentiment Analysis Tool

This tool analyzes the sentiment of comments on a YouTube video using the YouTube Data API and NLTK's VADER sentiment analyzer.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/youtube-sentiment-analysis.git
   cd youtube-sentiment-analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up a YouTube Data API key:
   - Go to the [Google Developers Console](https://console.developers.google.com/)
   - Create a new project or select an existing one
   - Enable the YouTube Data API v3
   - Create credentials (API key)
   - Set the API key as an environment variable:
     ```
     export YOUTUBE_API_KEY='your-api-key-here'
     ```

## Usage

Run the script with a YouTube video ID as an argument:

```
python main.py VIDEO_ID
```

Replace `VIDEO_ID` with the ID of the YouTube video you want to analyze. The video ID is the value of the 'v' parameter in the video's URL. For example, if the video URL is `https://www.youtube.com/watch?v=dQw4w9WgXcQ`, the video ID would be `dQw4w9WgXcQ`.

## Output

The tool will print the sentiment analysis results, including:
- The total number of comments analyzed
- The percentage of positive, neutral, and negative comments

## Limitations

- The tool is limited by the YouTube API's quota and rate limits.
- Sentiment analysis is performed using a pre-trained model and may not always accurately capture the intended sentiment, especially for complex or nuanced comments.
- The tool currently analyzes only the top-level comments and does not include replies.

## Screenshots

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/d246bfb6-5efe-45c7-939a-60258af4e20b" />

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/5df9da15-82ab-4998-9319-28a41ba9df30" />

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/c711582e-1fe7-4767-9e6d-ca57babfb922" />


## License

This project is open source and available under the [MIT License](LICENSE).

