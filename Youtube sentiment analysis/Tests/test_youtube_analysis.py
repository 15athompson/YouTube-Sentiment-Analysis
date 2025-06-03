import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from io import StringIO
import json
import sys
import os

# Add the parent directory to the Python path to import the main script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from youtube_analysis import (
    preprocess_text,
    analyze_sentiment,
    visualize_sentiment,
    generate_wordcloud,
    analyze_top_words,
    perform_topic_modeling,
    compare_videos
)

class TestYoutubeAnalysis(unittest.TestCase):

    def setUp(self):
        self.sample_comment = "This video is great! I learned a lot."
        self.sample_df = pd.DataFrame({
            'comment': ['Great video!', 'I disagree.', 'Interesting topic.'],
            'sentiment_scores': [{'compound': 0.8}, {'compound': -0.4}, {'compound': 0.2}],
            'sentiment': ['Positive', 'Negative', 'Positive'],
            'subjectivity': [0.9, 0.7, 0.5],
            'word_count': [2, 2, 2],
            'preprocessed_words': [['great', 'video'], ['disagree'], ['interesting', 'topic']]
        })

    def test_preprocess_text(self):
        result = preprocess_text(self.sample_comment)
        self.assertEqual(result, ['video', 'great', 'learn', 'lot'])

    def test_analyze_sentiment(self):
        result = analyze_sentiment(self.sample_comment)
        self.assertIsInstance(result, dict)
        self.assertIn('sentiment', result)
        self.assertIn('subjectivity', result)
        self.assertIn('word_count', result)

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_sentiment(self, mock_savefig):
        visualize_sentiment(self.sample_df, 'test_video_id')
        mock_savefig.assert_called_once_with('output/test_video_id/sentiment_distribution.png')

    @patch('wordcloud.WordCloud.generate')
    @patch('matplotlib.pyplot.savefig')
    def test_generate_wordcloud(self, mock_savefig, mock_generate):
        generate_wordcloud(self.sample_df, 'test_video_id')
        mock_generate.assert_called_once()
        mock_savefig.assert_called_once_with('output/test_video_id/wordcloud.png')

    @patch('matplotlib.pyplot.savefig')
    def test_analyze_top_words(self, mock_savefig):
        result = analyze_top_words(self.sample_df, 'test_video_id', top_n=2)
        self.assertEqual(len(result), 2)
        mock_savefig.assert_called_once_with('output/test_video_id/top_words.png')

    def test_perform_topic_modeling(self):
        result = perform_topic_modeling(self.sample_df, 'test_video_id', num_topics=2)
        self.assertEqual(len(result), 2)
        for topic in result:
            self.assertTrue(topic.startswith('Topic'))

    @patch('matplotlib.pyplot.savefig')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_compare_videos(self, mock_open, mock_savefig):
        all_results = {
            'video1': self.sample_df,
            'video2': self.sample_df
        }
        compare_videos(all_results)
        mock_savefig.assert_called_once_with('output/video_comparison.png')
        mock_open.assert_called_once_with('output/video_comparison.csv', 'w')

if __name__ == '__main__':
    unittest.main()