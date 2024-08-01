import tkinter as tk
from tkinter import ttk, messagebox
import threading
from youtube_data_extractor import YouTubeDataExtractor
from sentiment_analyzer import SentimentAnalyzer
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class YouTubeSentimentGUI:
    def __init__(self, master):
        self.master = master
        master.title("YouTube Sentiment Analysis Tool")
        master.geometry("800x600")

        # Create and set up widgets
        self.setup_widgets()

        # Initialize extractor and analyzer
        self.extractor = YouTubeDataExtractor()
        self.analyzer = SentimentAnalyzer()

    def create_sentiment_trend_graph(self, all_results):
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)

        for video_id, publish_date, _, _, _, _, sentiment_data in all_results:
            timestamps = sentiment_data['timestamps']
            scores = sentiment_data['scores']
            moving_average = sentiment_data['moving_average']

            ax.scatter(timestamps, scores, alpha=0.5, s=10, label=f'Video {video_id}')
            ax.plot(timestamps[len(timestamps)-len(moving_average):], moving_average, label=f'Moving Avg {video_id}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Trend Over Time')
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def setup_widgets(self):
        # Analysis type selection
        self.analysis_type = tk.StringVar(value="video")
        ttk.Radiobutton(self.master, text="Analyze Video", variable=self.analysis_type, value="video", command=self.update_input_label).pack()
        ttk.Radiobutton(self.master, text="Analyze Channel", variable=self.analysis_type, value="channel", command=self.update_input_label).pack()

        # ID input
        self.id_label = ttk.Label(self.master, text="Enter YouTube Video ID:")
        self.id_label.pack(pady=10)
        self.id_entry = ttk.Entry(self.master, width=50)
        self.id_entry.pack()

        # Max comments input
        ttk.Label(self.master, text="Max Comments per Video:").pack(pady=10)
        self.max_comments_entry = ttk.Entry(self.master, width=10)
        self.max_comments_entry.insert(0, "500")  # Default value
        self.max_comments_entry.pack()

        # Max videos input (for channel analysis)
        self.max_videos_label = ttk.Label(self.master, text="Max Videos to Analyze:")
        self.max_videos_entry = ttk.Entry(self.master, width=10)
        self.max_videos_entry.insert(0, "10")  # Default value

        # Analyze button
        self.analyze_button = ttk.Button(self.master, text="Analyze", command=self.start_analysis)
        self.analyze_button.pack(pady=20)

        # Progress bar
        self.progress = ttk.Progressbar(self.master, length=400, mode='indeterminate')
        self.progress.pack(pady=10)

        # Results display
        self.results_text = tk.Text(self.master, height=10, width=80)
        self.results_text.pack(pady=10)

        # Sentiment trend graph
        self.graph_frame = ttk.Frame(self.master)
        self.graph_frame.pack(pady=10, expand=True, fill=tk.BOTH)

    def update_input_label(self):
        if self.analysis_type.get() == "video":
            self.id_label.config(text="Enter YouTube Video ID:")
            self.max_videos_label.pack_forget()
            self.max_videos_entry.pack_forget()
        else:
            self.id_label.config(text="Enter YouTube Channel ID:")
            self.max_videos_label.pack()
            self.max_videos_entry.pack()

    def start_analysis(self):
        # Disable button and show progress bar
        self.analyze_button.config(state='disabled')
        self.progress.start()

        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Get input values
        id_value = self.id_entry.get().strip()
        max_comments = int(self.max_comments_entry.get())

        # Validate input
        if not self.is_valid_id(id_value):
            messagebox.showerror("Error", f"Invalid YouTube {'Video' if self.analysis_type.get() == 'video' else 'Channel'} ID")
            self.reset_ui()
            return

        # Get max videos for channel analysis
        max_videos = int(self.max_videos_entry.get()) if self.analysis_type.get() == "channel" else 1

        # Run analysis in a separate thread
        threading.Thread(target=self.run_analysis, args=(id_value, max_comments, max_videos), daemon=True).start()

    def run_analysis(self, id_value, max_comments, max_videos):
        try:
            if self.analysis_type.get() == "video":
                video_ids = [(id_value, None)]  # Add a dummy timestamp for consistency
            else:
                video_ids = self.extractor.get_channel_videos(id_value, max_videos)

            all_results = []
            for video_id, publish_date in video_ids:
                comments = self.extractor.extract_comments(video_id, max_comments)
                analyzed_comments = self.analyzer.analyze_comments(comments)
                sentiment_data = self.analyzer.prepare_sentiment_data(analyzed_comments)
                overall_sentiment, overall_score = self.analyzer.get_overall_sentiment(analyzed_comments)
                total_comments = len(comments)
                total_replies = sum(len(comment['replies']) for comment in comments)
                all_results.append((video_id, publish_date, total_comments, total_replies, overall_sentiment, overall_score, sentiment_data))

            self.display_results(id_value, all_results)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.reset_ui()

    def display_results(self, id_value, all_results):
        result_text = f"Sentiment Analysis Results for {'Video' if self.analysis_type.get() == 'video' else 'Channel'} ID: {id_value}\n\n"
        
        for video_id, publish_date, total_comments, total_replies, overall_sentiment, overall_score, sentiment_data in all_results:
            result_text += f"Video ID: {video_id}\n"
            if publish_date:
                result_text += f"Publish Date: {publish_date}\n"
            result_text += f"Total top-level comments analyzed: {total_comments}\n"
            result_text += f"Total replies analyzed: {total_replies}\n"
            result_text += f"Overall Sentiment: {overall_sentiment.capitalize()} (Score: {overall_score:.2f})\n\n"

        self.results_text.insert(tk.END, result_text)
        
        # Create and display the sentiment trend graph
        self.create_sentiment_trend_graph(all_results)

    def reset_ui(self):
        self.analyze_button.config(state='normal')
        self.progress.stop()

    @staticmethod
    def is_valid_id(id_value):
        # Basic validation for YouTube video ID or channel ID format
        return re.match(r'^[a-zA-Z0-9_-]{11,}$', id_value) is not None

if __name__ == "__main__":
    root = tk.Tk()
    app = YouTubeSentimentGUI(root)
    root.mainloop()