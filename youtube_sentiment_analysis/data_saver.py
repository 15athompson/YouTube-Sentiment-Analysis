import csv
import os
import sqlite3
import json

def save_to_csv(filename, sentiment_results, emoji_analysis):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['comment', 'translated', 'language', 'date', 'likes', 'compound', 'positive', 'neutral', 'negative', 'sentiment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in sentiment_results:
            writer.writerow(result)
        
        writer.writerow({})  # Empty row for separation
        writer.writerow({'comment': 'Top 10 Emojis'})
        for emoji, count in emoji_analysis:
            writer.writerow({'comment': emoji, 'compound': count})
    
    print(f"Results saved to {os.path.abspath(filename)}")

def save_to_database(db_file, results):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS video_analysis
                      (video_id TEXT PRIMARY KEY, title TEXT, data JSON)''')

    for result in results:
        video_id = result['video_id']
        title = result['title']
        data = json.dumps(result)

        cursor.execute("INSERT OR REPLACE INTO video_analysis (video_id, title, data) VALUES (?, ?, ?)",
                       (video_id, title, data))

    conn.commit()
    conn.close()

    print(f"Results saved to database: {db_file}")