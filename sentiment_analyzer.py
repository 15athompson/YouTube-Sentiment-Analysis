import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
from datetime import datetime

class SentimentAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model.to(self.device)
        self.model.eval()

    def analyze_comment(self, comment):
        inputs = self.tokenizer(comment, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = probabilities[0][1].item()  # Probability of positive sentiment
        
        if sentiment_score > 0.6:
            return 'positive', sentiment_score
        elif sentiment_score < 0.4:
            return 'negative', sentiment_score
        else:
            return 'neutral', sentiment_score

    def analyze_comments(self, comments):
        results = []
        for comment in comments:
            sentiment, score = self.analyze_comment(comment['text'])
            comment_data = {
                'sentiment': sentiment,
                'score': score,
                'timestamp': datetime.fromisoformat(comment['publish_date'].rstrip('Z')),
                'replies': []
            }
            for reply in comment['replies']:
                reply_sentiment, reply_score = self.analyze_comment(reply['text'])
                reply_data = {
                    'sentiment': reply_sentiment,
                    'score': reply_score,
                    'timestamp': datetime.fromisoformat(reply['publish_date'].rstrip('Z'))
                }
                comment_data['replies'].append(reply_data)
            results.append(comment_data)
        return results

    def prepare_sentiment_data(self, analyzed_comments):
        all_data = []
        for comment in analyzed_comments:
            all_data.append((comment['timestamp'], comment['score']))
            for reply in comment['replies']:
                all_data.append((reply['timestamp'], reply['score']))
        
        # Sort data by timestamp
        all_data.sort(key=lambda x: x[0])
        
        timestamps, scores = zip(*all_data)
        
        return {
            'timestamps': timestamps,
            'scores': scores,
            'moving_average': self.calculate_moving_average(scores)
        }

    @staticmethod
    def calculate_moving_average(scores, window=10):
        return np.convolve(scores, np.ones(window), 'valid') / window

    def get_overall_sentiment(self, analyzed_comments):
        total_score = 0
        total_count = 0
        for comment in analyzed_comments:
            total_score += comment['score']
            total_count += 1
            for reply in comment['replies']:
                total_score += reply['score']
                total_count += 1
        
        average_score = total_score / total_count if total_count > 0 else 0.5
        
        if average_score > 0.6:
            return 'positive', average_score
        elif average_score < 0.4:
            return 'negative', average_score
        else:
            return 'neutral', average_score