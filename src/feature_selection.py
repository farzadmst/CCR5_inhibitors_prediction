import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def feature_selection(input_csv="data/HIVfinaldataset.csv", output_csv="data/HIVfinaldataset_selected_features.csv"):
    """Performs feature selection using VarianceThreshold and saves the dataset with selected features."""

    # Load the merged dataset
    dataset = pd.read_csv(input_csv)

    # Split dataset into features (X) and target (y)
    X = dataset.drop(columns=['pIC50'])  # Features (dropping the target column)

    # Apply VarianceThreshold to remove features with low variance
    selection = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))  # Variance threshold of 0.16
    X_selected = selection.fit_transform(X)

    # Get the selected feature names
    selected_feature_names = X.columns[selection.get_support()]

    # Print the selected features
    print("Selected features based on VarianceThreshold:")
    print(selected_feature_names)

    # Update the dataset with only selected features
    selected_dataset = pd.DataFrame(X_selected, columns=selected_feature_names)
    selected_dataset['pIC50'] = dataset['pIC50']  # Add the target column back

    # Save the dataset with selected features to a CSV file
    selected_dataset.to_csv(output_csv, index=False)
    print(f"Dataset with selected features saved to: {output_csv}")


if __name__ == "__main__":
    feature_selection()
