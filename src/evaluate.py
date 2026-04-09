import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return mae, mse, r2

def save_metrics(mae, mse, r2):
    os.makedirs("reports", exist_ok=True)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "R2": r2
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

def load_metrics():
    try:
        with open("reports/metrics.json", "r") as f:
            metrics = json.load(f)

        print("\n📊 Saved Model Metrics:")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"MSE: {metrics['MSE']:.2f}")
        print(f"R2 Score: {metrics['R2']:.2f}")

    except FileNotFoundError:
        print("❌ No metrics found. Train model first.")