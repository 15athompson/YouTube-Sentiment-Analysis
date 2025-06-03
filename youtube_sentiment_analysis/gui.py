import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QDateEdit, QSpinBox, QCheckBox
from PyQt5.QtCore import QThread, pyqtSignal, QDate
from PyQt5.QtWebEngineWidgets import QWebEngineView
import asyncio
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from main import process_video
from cache_manager import CacheManager

class AnalysisThread(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, video_id, max_comments, start_date, end_date, min_likes):
        super().__init__()
        self.video_id = video_id
        self.max_comments = max_comments
        self.start_date = start_date
        self.end_date = end_date
        self.min_likes = min_likes

    def run(self):
        cache_manager = CacheManager()
        result = asyncio.run(process_video(self.video_id, self.max_comments, cache_manager, self.start_date, self.end_date, self.min_likes))
        self.finished.emit(result)

class YouTubeSentimentAnalyzerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        self.video_id_input = QLineEdit()
        self.video_id_input.setPlaceholderText("Enter YouTube Video ID")
        input_layout.addWidget(self.video_id_input)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.start_analysis)
        input_layout.addWidget(self.analyze_button)

        layout.addLayout(input_layout)

        filter_layout = QHBoxLayout()
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addYears(-1))
        filter_layout.addWidget(QLabel("Start Date:"))
        filter_layout.addWidget(self.start_date)

        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        filter_layout.addWidget(QLabel("End Date:"))
        filter_layout.addWidget(self.end_date)

        self.min_likes = QSpinBox()
        self.min_likes.setRange(0, 1000000)
        filter_layout.addWidget(QLabel("Min Likes:"))
        filter_layout.addWidget(self.min_likes)

        layout.addLayout(filter_layout)

        self.use_cache = QCheckBox("Use Cache")
        self.use_cache.setChecked(True)
        layout.addWidget(self.use_cache)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.plot_view = QWebEngineView()
        layout.addWidget(self.plot_view)

        self.setLayout(layout)
        self.setWindowTitle("YouTube Sentiment Analyzer")
        self.setGeometry(300, 300, 800, 600)

    def start_analysis(self):
        video_id = self.video_id_input.text()
        if not video_id:
            self.result_text.setText("Please enter a valid YouTube Video ID")
            return

        self.analyze_button.setEnabled(False)
        self.result_text.setText("Analyzing...")

        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")
        min_likes = self.min_likes.value()

        self.analysis_thread = AnalysisThread(video_id, 500, start_date, end_date, min_likes)
        self.analysis_thread.finished.connect(self.display_results)
        self.analysis_thread.start()

    def display_results(self, result):
        if result:
            output = f"Video: {result['title']} ({result['video_id']})\n"
            output += f"Title sentiment: {result['title_sentiment']['sentiment']} (compound: {result['title_sentiment']['compound']:.2f})\n"
            output += f"Description sentiment: {result['description_sentiment']['sentiment']} (compound: {result['description_sentiment']['compound']:.2f})\n\n"

            sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
            for comment_result in result['sentiment_results']:
                sentiment_counts[comment_result['sentiment']] += 1

            total_comments = sum(sentiment_counts.values())
            output += "Comment Sentiment Distribution:\n"
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total_comments) * 100
                output += f"{sentiment}: {count} ({percentage:.1f}%)\n"

            output += "\nTop 10 Emojis:\n"
            for emoji, count in result['emoji_analysis']:
                output += f"{emoji}: {count}\n"

            self.result_text.setText(output)

            self.plot_interactive_charts(result)
        else:
            self.result_text.setText("An error occurred during analysis.")

        self.analyze_button.setEnabled(True)

    def plot_interactive_charts(self, result):
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Sentiment Distribution", "Word Cloud", "Sentiment Over Time", "Sentiment Trend"))

        # Sentiment Distribution
        compound_scores = [r['compound'] for r in result['sentiment_results']]
        fig.add_trace(go.Histogram(x=compound_scores, nbinsx=20, marker_color='blue'), row=1, col=1)
        fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=len(compound_scores), line=dict(color="red", width=2), row=1, col=1)

        # Word Cloud
        word_cloud_data = go.Scatter(x=[0], y=[0], mode="text", text=["Word Cloud"], textfont=dict(size=30))
        fig.add_trace(word_cloud_data, row=1, col=2)

        # Sentiment Over Time
        dates = [r['date'] for r in result['sentiment_results']]
        compound_scores = [r['compound'] for r in result['sentiment_results']]
        fig.add_trace(go.Scatter(x=dates, y=compound_scores, mode='markers', marker=dict(color=compound_scores, colorscale='RdYlGn', showscale=True)), row=2, col=1)
        fig.add_shape(type="line", x0=min(dates), y0=0, x1=max(dates), y1=0, line=dict(color="red", width=2), row=2, col=1)

        # Sentiment Trend
        trend_dates = [r['date'] for r in result['sentiment_trend']['dates']]
        moving_average = result['sentiment_trend']['moving_average']
        fig.add_trace(go.Scatter(x=trend_dates, y=moving_average, mode='lines', line=dict(color='blue')), row=2, col=2)
        fig.add_shape(type="line", x0=min(trend_dates), y0=0, x1=max(trend_dates), y1=0, line=dict(color="red", width=2), row=2, col=2)

        fig.update_layout(height=800, showlegend=False)
        self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

def run_gui():
    app = QApplication(sys.argv)
    ex = YouTubeSentimentAnalyzerGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_gui()
 