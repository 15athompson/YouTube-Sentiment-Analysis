import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

class PopularityPredictor:
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(self, data):
        # Assuming data is a DataFrame with features and target
        X = data.drop('popularity', axis=1)  # Features
        y = data['popularity']  # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate the model
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f'Model trained with MSE: {mse}')

    def predict(self, features):
        return self.model.predict([features])

    def save_model(self, filename):
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)