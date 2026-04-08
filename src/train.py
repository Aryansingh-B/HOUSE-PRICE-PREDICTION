from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os
from src.evaluate import evaluate_model

def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    evaluate_model(y_test, y_pred)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\n✅ Model trained, evaluated & saved")
    return model