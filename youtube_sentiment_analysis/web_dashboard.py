from flask import Flask, render_template, request, jsonify
from cache_manager import CacheManager
from main import process_video
import asyncio

app = Flask(__name__)
cache_manager = CacheManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_id = request.form['video_id']
    max_comments = int(request.form.get('max_comments', 500))
    
    # Run the analysis asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(process_video(video_id, max_comments, cache_manager))
    loop.close()

    if result:
        return jsonify({"status": "success", "video_id": video_id})
    else:
        return jsonify({"status": "error", "message": "Analysis failed"}), 400

@app.route('/results/<video_id>')
def get_results(video_id):
    results = cache_manager.get_results(video_id)
    if results:
        # Ensure all necessary data is included
        response = {
            "video_id": results["video_id"],
            "title": results["title"],
            "sentiment_results": results["sentiment_results"],
            "topics": results["topics"],
            "named_entities": results["named_entities"],
            "sentiment_aspects": results["sentiment_aspects"],
            "transcript_sentiment": results["transcript_sentiment"],
            "audio_sentiment": results["audio_sentiment"],
            "sentiment_trend": results["sentiment_trend"]
        }
        return jsonify(response)
    else:
        return jsonify({"status": "error", "message": "Results not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)