import argparse
from youtube_data_extractor import YouTubeDataExtractor
from sentiment_analyzer import SentimentAnalyzer

def count_comments_and_replies(comments):
    total_comments = len(comments)
    total_replies = sum(len(comment['replies']) for comment in comments)
    return total_comments, total_replies

def main():
    parser = argparse.ArgumentParser(description="YouTube Sentiment Analysis Tool")
    parser.add_argument("video_id", help="YouTube video ID to analyze")
    parser.add_argument("--max_comments", type=int, default=500, help="Maximum number of comments to analyze (including replies)")
    args = parser.parse_args()

    # Extract YouTube data
    extractor = YouTubeDataExtractor()
    comments = extractor.extract_comments(args.video_id, args.max_comments)

    # Count comments and replies
    total_comments, total_replies = count_comments_and_replies(comments)

    # Analyze sentiment
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_comments(comments)

    # Print results
    print(f"\nSentiment Analysis Results for Video ID: {args.video_id}")
    print(f"Total top-level comments analyzed: {total_comments}")
    print(f"Total replies analyzed: {total_replies}")
    print(f"Total comments and replies analyzed: {total_comments + total_replies}")
    print("\nSentiment Breakdown:")
    print(f"Positive: {results['positive']:.2f}%")
    print(f"Neutral: {results['neutral']:.2f}%")
    print(f"Negative: {results['negative']:.2f}%")

if __name__ == "__main__":
    main()