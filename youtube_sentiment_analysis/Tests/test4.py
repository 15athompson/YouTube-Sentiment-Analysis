import textblob
from textblob import TextBlob

def analyze_sentiment(text):
  blob = TextBlob(text)
  sentiment = blob.sentiment
  return sentiment.polarity, sentiment.subjectivity   


# Example usage
text = "This is a great video!"
polarity, subjectivity = analyze_sentiment(text)
print("Polarity:", polarity)
print("Subjectivity:", subjectivity)