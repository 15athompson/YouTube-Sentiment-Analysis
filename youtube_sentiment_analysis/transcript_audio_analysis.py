from youtube_transcript_api import YouTubeTranscriptApi
from pydub import AudioSegment
import speech_recognition as sr
from textblob import TextBlob
import requests
import os

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {str(e)}")
        return None

def download_audio(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_file = f"{video_id}.mp3"
    
    try:
        os.system(f"youtube-dl -x --audio-format mp3 -o {output_file} {url}")
        return output_file
    except Exception as e:
        print(f"Error downloading audio for video {video_id}: {str(e)}")
        return None

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return None

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def analyze_transcript_and_audio(video_id):
    transcript = get_video_transcript(video_id)
    transcript_sentiment = analyze_sentiment(transcript) if transcript else None
    
    audio_file = download_audio(video_id)
    if audio_file:
        audio_transcript = transcribe_audio(audio_file)
        audio_sentiment = analyze_sentiment(audio_transcript) if audio_transcript else None
        os.remove(audio_file)  # Clean up the downloaded audio file
    else:
        audio_transcript = None
        audio_sentiment = None
    
    return {
        "transcript": transcript,
        "transcript_sentiment": transcript_sentiment,
        "audio_transcript": audio_transcript,
        "audio_sentiment": audio_sentiment
    }