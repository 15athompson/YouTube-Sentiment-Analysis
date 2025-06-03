import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import string
from datetime import datetime
import os
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

def visualize_sentiment(sentiment_results):
    compound_scores = [result['compound'] for result in sentiment_results]
    
    plt.figure(figsize=(10, 6))
    plt.hist(compound_scores, bins=20, edgecolor='black')
    plt.title("Sentiment Distribution of Comments from YouTube, Twitter, and Reddit")
    plt.xlabel("Compound Sentiment Score")
    plt.ylabel("Frequency")
    plt.axvline(x=0, color='r', linestyle='--', label='Neutral')
    plt.legend()
    plt.show()

    # Pie chart for sentiment distribution
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for result in sentiment_results:
        sentiment_counts[result['sentiment']] += 1

    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%')
    plt.title("Sentiment Distribution")
    plt.show()

def create_word_cloud(comments):
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    # Combine all comments and remove punctuation
    text = ' '.join([comment['text'] for comment in comments])
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of YouTube Comments')
    plt.show()

def visualize_sentiment_over_time(sentiment_results):
    dates = [datetime.fromisoformat(result['date'].replace('Z', '+00:00')) for result in sentiment_results]
    compound_scores = [result['compound'] for result in sentiment_results]

    plt.figure(figsize=(12, 6))
    plt.scatter(dates, compound_scores, alpha=0.5)
    plt.title("Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Compound Sentiment Score")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_sentiment_trend(sentiment_trend):
    """
    Visualize the average sentiment trend over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(sentiment_trend['dates'], sentiment_trend['average_scores'], marker='o')
    plt.title("Average Sentiment Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Compound Sentiment Score")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_named_entities(named_entities):
    """
    Create a bar chart of the top 10 named entities.
    """
    sorted_entities = sorted(named_entities.items(), key=lambda x: x[1], reverse=True)[:10]
    entities, counts = zip(*sorted_entities)

    plt.figure(figsize=(12, 6))
    plt.bar(entities, counts)
    plt.title("Top 10 Named Entities")
    plt.xlabel("Entity")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("named_entities_chart.png")
    plt.close()

def visualize_sentiment_aspects(sentiment_aspects):
    """
    Create a horizontal bar chart of sentiment aspects.
    """
    aspects = list(sentiment_aspects.keys())
    sentiments = list(sentiment_aspects.values())

    fig = go.Figure(go.Bar(
        y=aspects,
        x=sentiments,
        orientation='h',
        marker_color=sentiments,
        marker_colorscale='RdYlGn',
        marker_colorbar_title='Sentiment'
    ))

    fig.update_layout(
        title="Sentiment Aspects",
        xaxis_title="Sentiment Score",
        yaxis_title="Aspect",
        height=max(600, len(aspects) * 25)  # Adjust height based on number of aspects
    )

    fig.write_html("sentiment_aspects_chart.html")

def visualize_transcript_sentiment(transcript_sentiment):
    """
    Create a gauge chart for transcript sentiment.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = transcript_sentiment,
        title = {'text': "Transcript Sentiment"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0.5], 'color': "yellow"},
                {'range': [0.5, 1], 'color': "green"}
            ]
        }
    ))
    fig.write_html("transcript_sentiment_gauge.html")

def visualize_audio_sentiment(audio_sentiment):
    """
    Create a gauge chart for audio sentiment.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = audio_sentiment,
        title = {'text': "Audio Sentiment"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0.5], 'color': "yellow"},
                {'range': [0.5, 1], 'color': "green"}
            ]
        }
    ))
    fig.write_html("audio_sentiment_gauge.html")

def visualize_sentiment_comparison(comment_sentiment, transcript_sentiment, audio_sentiment):
    """
    Create a bar chart comparing comment, transcript, and audio sentiment.
    """
    categories = ['Comments', 'Transcript', 'Audio']
    sentiments = [comment_sentiment, transcript_sentiment, audio_sentiment]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, sentiments)
    plt.title("Sentiment Comparison: Comments vs Content")
    plt.ylabel("Sentiment Score")
    plt.ylim(-1, 1)
    
    # Color the bars based on sentiment
    for bar, sentiment in zip(bars, sentiments):
        if sentiment < -0.5:
            bar.set_color('red')
        elif sentiment < 0.5:
            bar.set_color('yellow')
        else:
            bar.set_color('green')
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("sentiment_comparison.png")
    plt.close()

def save_visualizations(results):
    for result in results:
        video_id = result['video_id']
        os.makedirs(f"visualizations/{video_id}", exist_ok=True)

        plt.figure(figsize=(10, 6))
        visualize_sentiment(result['sentiment_results'])
        plt.savefig(f"visualizations/{video_id}/sentiment_distribution.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        create_word_cloud([comment['text'] for comment in result['sentiment_results']])
        plt.savefig(f"visualizations/{video_id}/word_cloud.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        visualize_sentiment_over_time(result['sentiment_results'])
        plt.savefig(f"visualizations/{video_id}/sentiment_over_time.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        visualize_sentiment_trend(result['sentiment_trend'])
        plt.savefig(f"visualizations/{video_id}/sentiment_trend.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        visualize_topic_distribution(result['topics'])
        plt.savefig(f"visualizations/{video_id}/topic_distribution.png")
        plt.close()

        visualize_named_entities(result['named_entities'])
        visualize_sentiment_aspects(result['sentiment_aspects'])

        # New visualizations for transcript and audio sentiment
        visualize_transcript_sentiment(result['transcript_sentiment'])
        visualize_audio_sentiment(result['audio_sentiment'])

        # Sentiment comparison visualization
        average_comment_sentiment = sum(r['compound'] for r in result['sentiment_results']) / len(result['sentiment_results'])
        visualize_sentiment_comparison(average_comment_sentiment, result['transcript_sentiment'], result['audio_sentiment'])

    print("Visualizations saved in the 'visualizations' directory.")

def visualize_topic_distribution(topics):
    topic_ids = [topic['topic_id'] for topic in topics]
    word_counts = [len(topic['words']) for topic in topics]

    plt.figure(figsize=(12, 6))
    plt.bar(topic_ids, word_counts)
    plt.title("Topic Distribution")
    plt.xlabel("Topic ID")
    plt.ylabel("Number of Words")
    plt.xticks(topic_ids)
    plt.tight_layout()
    plt.show()