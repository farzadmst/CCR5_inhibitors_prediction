import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_absolute_error
import numpy as np

def train_model(input_csv="data/HIVfinaldataset_selected_features.csv", model_output="data/rf_trained_model.pkl"):
    """Trains a Random Forest model using cross-validation and saves the best model."""

    # Load the dataset with selected features
    dataset = pd.read_csv(input_csv)

    # Split dataset into features (X) and target (y)
    X = dataset.drop(columns=['pIC50'])  # Features (dropping the target column)
    y = dataset['pIC50']  # Target (pIC50)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')

    # Print the cross-validation results
    print("Cross-validation MAE scores:", -cv_scores)
    print("Average CV MAE:", -cv_scores.mean())

    # Train the Random Forest model on the entire training set
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Calculate the Mean Absolute Error (MAE) for the test set
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test set Mean Absolute Error: {mae}")

    # Save the trained RandomForest model
    joblib.dump(rf_model, model_output)
    print(f"Trained Random Forest model saved to: {model_output}")

if __name__ == "__main__":
    train_model()
