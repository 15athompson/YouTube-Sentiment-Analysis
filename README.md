# YouTube-Sentiment-Analysis

Description:
This script is for extracting comments from YouTube videos using the YouTube Data API. The code imports necessary libraries, configures API access with an API key, and defines functions to retrieve comments for a specific video. Here is a summary of the key components and functionalities of the code:

Imports: The code imports essential libraries, including os for operating system-related functions, googleapiclient.discovery for interacting with the YouTube Data API, and dotenv for loading environment variables.

API Key Configuration: The code loads an API key from environment variables to authenticate and access the YouTube Data API.

Comment Retrieval Function: The get_comments function fetches comments from a specified YouTube video using the YouTube Data API. It paginates through the comments to ensure that all comments are retrieved.

Main Function: The main function configures the YouTube Data API client with the provided API key and uses the get_comments function to extract comments for a given video.

Video Comment Retrieval Function: The get_video_comments function serves as an entry point for retrieving comments for a specific video. It calls the main function with the video ID and API key, and returns the comments as a result.
