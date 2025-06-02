import pandas as pd
from popularity_predictor import PopularityPredictor

def train_model():
    # Load historical data
    data = pd.read_csv('historical_video_data.csv')

    # Initialize the predictor
    predictor = PopularityPredictor()

    # Train the model
    predictor.train(data)

    # Save the model
    predictor.save_model('popularity_model.pkl')

if __name__ == "__main__":
    train_model()