# CCR5 (HIV receptor) Inhibitor Prediction with Random Forest

This repository contains a machine learning project aimed at predicting the pIC50 values of CCR5 inhibitors based on molecular features. The model uses Random Forest Regressor for predicting inhibitory activity from molecular fingerprints. The project includes feature selection, data preprocessing, and model training to build a predictive model for HIV inhibitors.

## Features

- **Data Preprocessing**: Merges molecular descriptors (fingerprints) with experimental pIC50 values.
- **Feature Selection**: Implements variance threshold-based feature selection to reduce irrelevant features.
- **Model Training**: Trains a Random Forest Regressor using cross-validation to estimate model performance.
- **Model Saving**: Saves the trained model for future use with `joblib`.
- **Evaluation**: Evaluates the model using Mean Absolute Error (MAE).

## Project Workflow

1. **Merge Data**: The molecular fingerprint dataset and the pIC50 values are merged into a final dataset for training.
2. **Feature Selection**: A variance threshold filter is applied to select relevant features.
3. **Train Model**: The Random Forest Regressor is trained with cross-validation on the training set.
4. **Model Evaluation**: The model is evaluated using MAE on the test set.
5. **Model Save**: The trained model is saved to a file for future predictions.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/HIV-inhibitor-prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd HIV-inhibitor-prediction
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Merging Data**: The `merge_data.py` script merges the fingerprint data and pIC50 values into one dataset.

    ```bash
    python merge_data.py
    ```

2. **Feature Selection**: The `feature_selection.py` script applies variance threshold-based feature selection to the dataset.

    ```bash
    python feature_selection.py
    ```

3. **Training the Model**: The `train_model.py` script trains a Random Forest model with cross-validation and saves the best model.

    ```bash
    python train_model.py
    ```

4. **Predictions**: After training, you can use the saved model to make predictions on new data.

    ```python
    import joblib
    model = joblib.load('data/rf_trained_model.pkl')
    predictions = model.predict(X_new)  # Replace X_new with your new data
    ```

## Results

- The trained Random Forest model achieved a Mean Absolute Error (MAE) of 0.58 on the test set. This result indicates that the model's predictions are reasonably close to the actual values, but there is still room for improvement. While an MAE of 0.58 is not bad, further optimizations through hyperparameter tuning, feature engineering, or exploring other machine learning models could lead to better performance and more accurate predictions.

## Dependencies

- `pandas`: For data manipulation and merging.
- `numpy`: For numerical operations.
- `scikit-learn`: For machine learning and feature selection.
- `joblib`: For saving and loading the trained model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
