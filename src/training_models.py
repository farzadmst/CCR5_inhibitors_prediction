import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
import joblib
from sklearn.metrics import mean_absolute_error


def train_model(input_csv="data/HIVfinaldataset_selected_features.csv", model_output="data/best_trained_model.pkl"):
    """Train multiple models using LazyRegressor and selects the best performing one."""

    # Load the dataset with selected features
    dataset = pd.read_csv(input_csv)

    # Split dataset into features (X) and target (y)
    X = dataset.drop(columns=['pIC50'])  # Features (dropping the target column)
    y = dataset['pIC50']  # Target (pIC50)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize LazyRegressor to compare different models
    regressor = LazyRegressor()

    # Fit the models and get the results
    models, predictions = regressor.fit(X_train, X_test, y_train, y_test)

    # Print the results of the models tested
    print(models)

    # Find the model with the lowest MAE
    best_model_name = models.sort_values(by='MAE').index[0]
    best_model = regressor.models[best_model_name]

    # Train the best model
    best_model.fit(X_train, y_train)

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Calculate the Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Best model: {best_model_name}")
    print(f"Best model Mean Absolute Error: {mae}")

    # Save the trained best model
    joblib.dump(best_model, model_output)
    print(f"Trained best model saved to: {model_output}")


if __name__ == "__main__":
    train_model()
