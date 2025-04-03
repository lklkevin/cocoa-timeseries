import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import product
import joblib
import os

def load_and_prepare_data(train_path='train.csv', test_path='test.csv'):
    """Load and prepare the data, including handling missing values and deseasonalization."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Combine and sort
    df = pd.concat([train, test], ignore_index=True)
    df['MONTH'] = pd.to_datetime(df['MONTH'])
    df.sort_values('MONTH', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill missing TAVG and PRCP using monthly climatology
    df['month'] = df['MONTH'].dt.month
    df['TAVG'] = df['TAVG'].fillna(df.groupby('month')['TAVG'].transform('mean'))
    df['PRCP'] = df['PRCP'].fillna(df.groupby('month')['PRCP'].transform('mean'))
    df.drop(columns=['month'], inplace=True)

    # Deseasonalize TAVG and PRCP using STL
    df['TAVG'] = deseasonalize_stl(df['TAVG'])
    df['PRCP'] = deseasonalize_stl(df['PRCP'])

    return df

def deseasonalize_stl(series, period=12):
    """Deseasonalize a time series using STL decomposition."""
    stl = STL(series, period=period, robust=True)
    return series - stl.fit().seasonal

def engineer_features(df):
    """Create additional features from the data."""
    # Add lagged ICCO prices
    df['ICCO_lag1'] = df['ICCO_price'].shift(1)
    df['ICCO_lag2'] = df['ICCO_price'].shift(2)

    # Rolling statistics for ICCO_price
    df['ICCO_roll_mean'] = df['ICCO_price'].rolling(window=3).mean()
    df['ICCO_roll_std'] = df['ICCO_price'].rolling(window=3).std()

    # Differenced target as a feature
    df['ICCO_diff'] = df['ICCO_price'].diff()

    cols = ['CPI_world', 'Ghana_official', 'TAVG', 'PRCP', 'ICCO_price', 'MONTH',
            'ICCO_lag1', 'ICCO_lag2', 'ICCO_roll_mean', 'ICCO_roll_std', 'ICCO_diff']

    # Drop initial rows with NaNs from rolling and lagging
    df = df[cols]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def prepare_features_and_targets(df, features, target_column):
    """Scale features and prepare training/test splits."""
    # Create separate scalers for features and target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    df_scaled = df.copy()
    df_scaled[features] = feature_scaler.fit_transform(df[features])
    df_scaled[target_column] = target_scaler.fit_transform(df[[target_column]])

    # Restrict training to last 5 years
    latest_date = df_scaled['MONTH'].max()
    cutoff_date = latest_date - pd.DateOffset(years=5)
    df_train = df_scaled.copy()
    df_train = df_train[df_train['MONTH'] >= cutoff_date].iloc[:-4]
    df_test = df_scaled.iloc[-4:].copy()

    return df_train, df_test, feature_scaler, target_scaler

def create_windows(data, window_size, target_col, features):
    """Create sliding windows for LSTM input."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size][features].values)
        y.append(data.iloc[i+window_size][target_col])
    return np.array(X), np.array(y)

def build_model(window_size, feature_len, hidden_size=100, dropout_rate=0.2, layers=2):
    """Build and compile the LSTM model."""
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(window_size, feature_len)))
    for i in range(layers):
        return_seq = (i < layers - 1)
        model.add(LSTM(hidden_size, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')
    return model

def run_grid_search(df_train, features, target_column, window_sizes, hidden_sizes, layer_counts, epochs_list):
    """Perform grid search to find optimal hyperparameters."""
    best_config = None
    best_val_loss = float('inf')

    for window_size, hidden_size, layers, epochs in product(window_sizes, hidden_sizes, layer_counts, epochs_list):
        print(f"Training model: window_size={window_size}, hidden_size={hidden_size}, layers={layers}, epochs={epochs}")
        
        X_train_curr, y_train_curr = create_windows(df_train, window_size, target_column, features)
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train_curr, y_train_curr, test_size=0.2, shuffle=False)

        model = build_model(window_size, len(features), hidden_size=hidden_size, layers=layers)
        
        history = model.fit(
            X_train_sub, y_train_sub,
            epochs=epochs,
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=0
        )

        val_loss = min(history.history['val_loss'])
        print(f"Validation MAE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = (window_size, hidden_size, layers, epochs)

    print(f"\nBest configuration: window_size={best_config[0]}, hidden_size={best_config[1]}, layers={best_config[2]}, epochs={best_config[3]}")
    return best_config

def create_test_windows(train_data, test_data, window_size, features):
    """Create windows for test data predictions."""
    combined = pd.concat([train_data, test_data], ignore_index=True)
    X_test = []
    timestamps = []
    
    start_idx = len(train_data) - window_size
    for i in range(4):  # 4 test points
        window = combined.iloc[start_idx + i : start_idx + i + window_size][features].values
        X_test.append(window)
        timestamps.append(combined.iloc[start_idx + i + window_size]['MONTH'])
        
    return np.array(X_test), timestamps

def calculate_prediction_intervals(predictions, actual_values):
    """Calculate prediction intervals based on residuals."""
    residuals = actual_values - predictions
    std_residuals = np.std(residuals)
    z_score = 1.96  # 95% confidence interval

    upper_bound = predictions + z_score * std_residuals
    lower_bound = predictions - z_score * std_residuals
    return upper_bound, lower_bound

def plot_results(train_dates, ts_test, actual_values, y_true_test, y_pred, test_lower, test_upper):
    """Create visualization of results."""
    plt.figure(figsize=(12, 6))

    # Plot all actual values as a single line
    all_dates = pd.concat([train_dates, pd.Series(ts_test)])
    all_actual = np.concatenate([actual_values, y_true_test])
    plt.plot(all_dates, all_actual, label='Actual Values', color='blue', marker='o', markersize=4, alpha=0.7)

    # Plot test predictions with intervals
    plt.fill_between(ts_test, test_lower, test_upper, color='orange', alpha=0.1, label='Test 95% CI')
    plt.plot(ts_test, y_pred, label='Test Predictions', color='orange', linestyle='--', marker='x', markersize=6)

    plt.title('ICCO Price: Actual Values and Test Predictions with 95% Prediction Intervals', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('ICCO Price', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()

def print_metrics(actual_values, predictions, y_true_test, y_pred):
    """Print performance metrics for training and test sets."""
    print("\nTraining Set Performance Metrics:")
    train_mse = np.mean((actual_values - predictions)**2)
    train_mae = np.mean(np.abs(actual_values - predictions))
    train_rmse = np.sqrt(train_mse)
    train_mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
    print(f"Mean Squared Error (MSE): {train_mse:.4f}")
    print(f"Mean Absolute Error (MAE): {train_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {train_rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {train_mape:.2f}%")

    print("\nTest Set Performance Metrics:")
    test_mse = np.mean((y_true_test - y_pred)**2)
    test_mae = np.mean(np.abs(y_true_test - y_pred))
    test_rmse = np.sqrt(test_mse)
    test_mape = np.mean(np.abs((y_true_test - y_pred) / y_true_test)) * 100
    print(f"Mean Squared Error (MSE): {test_mse:.4f}")
    print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")

    print("\nPrediction Interval Statistics:")
    print(f"Test Set Standard Deviation of Residuals: {np.std(y_true_test - y_pred):.4f}")
    print(f"Test Set Prediction Standard Deviation: {np.std(y_pred):.4f}")

def save_model_and_scalers(model, feature_scaler, target_scaler, model_params, save_dir='saved_model'):
    """Save the trained model, scalers, and parameters."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save(f'{save_dir}/lstm_model.keras')
    joblib.dump(feature_scaler, f'{save_dir}/feature_scaler.pkl')
    joblib.dump(target_scaler, f'{save_dir}/target_scaler.pkl')
    joblib.dump(model_params, f'{save_dir}/model_params.pkl')

    print(f"\nModel and scalers saved successfully in '{save_dir}' directory")

def load_keras_model(model_path):
    """Load a saved Keras model from a .keras file.
    
    Args:
        model_path (str): Path to the .keras model file
        
    Returns:
        tf.keras.Model: The loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found")
    
    model = tf.keras.models.load_model(model_path)
    print(f"\nModel loaded successfully from '{model_path}'")
    return model

if __name__ == "__main__":
    # Define features and target
    features = ['CPI_world', 'Ghana_official', 'TAVG', 'PRCP', 
                'ICCO_lag1', 'ICCO_lag2', 'ICCO_roll_mean', 'ICCO_roll_std', 'ICCO_diff']
    target_column = 'ICCO_price'

    # Load and prepare data
    df = load_and_prepare_data()
    df = engineer_features(df)
    df_train, df_test, feature_scaler, target_scaler = prepare_features_and_targets(df, features, target_column)

    # Grid search parameters
    window_sizes = [6, 12, 24]
    hidden_sizes = [50, 100, 150]
    layer_counts = [1, 2, 3]
    epochs_list = [10, 25, 50]

    # Run grid search
    best_window_size, best_hidden_size, best_layers, best_epochs = run_grid_search(
        df_train, features, target_column, window_sizes, hidden_sizes, layer_counts, epochs_list
    )

    # OR use best parameters found from previous grid search
    # best_window_size = 24
    # best_hidden_size = 150
    # best_layers = 1
    # best_epochs = 25

    # Train final model
    X_train, y_train = create_windows(df_train, best_window_size, target_column, features)
    model = build_model(best_window_size, len(features), best_hidden_size, 0.2, best_layers)
    history = model.fit(
        X_train, y_train,
        epochs=best_epochs,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )

    # OR load model from saved file
    # model = load_keras_model('saved_model/lstm_model.keras')
    
    # Make predictions
    train_predictions_scaled = model.predict(X_train).flatten()
    train_predictions = target_scaler.inverse_transform(train_predictions_scaled.reshape(-1, 1)).flatten()
    actual_values = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

    X_test, ts_test = create_test_windows(df_train, df_test, best_window_size, features)
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true_test = df.loc[df['MONTH'].isin(ts_test), 'ICCO_price'].values

    # Create dates for training data
    train_dates = df_train['MONTH'].iloc[best_window_size:].reset_index(drop=True)

    # Calculate prediction intervals
    test_upper, test_lower = calculate_prediction_intervals(y_pred, y_true_test)

    # Visualize results
    plot_results(train_dates, ts_test, actual_values, y_true_test, y_pred, test_lower, test_upper)

    # Print metrics
    print_metrics(actual_values, train_predictions, y_true_test, y_pred)

    # Save model and scalers
    model_params = {
        'window_size': best_window_size,
        'hidden_size': best_hidden_size,
        'layers': best_layers,
        'features': features,
        'target_column': target_column
    }
    save_model_and_scalers(model, feature_scaler, target_scaler, model_params)