import json
import os

class CacheManager:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, video_id):
        return os.path.join(self.cache_dir, f"{video_id}.json")

    def save_results(self, video_id, results):
        cache_path = self.get_cache_path(video_id)
        with open(cache_path, 'w') as f:
            json.dump(results, f)

    def get_results(self, video_id):
        cache_path = self.get_cache_path(video_id)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None