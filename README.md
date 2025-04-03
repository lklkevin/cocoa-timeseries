# Cocoa Price Time Series Analysis and Forecasting

This project implements a comprehensive time series analysis and forecasting system for cocoa prices using multiple modeling approaches, including LSTM neural networks, ARMA-GARCH, and SARIMA models.

## Project Structure

- `PREPROCESS.R`: Data preprocessing script that combines and processes cocoa price data, weather data, and economic indicators
- `RNN.py`: Implementation of LSTM neural network for cocoa price forecasting
- `ARMA-GARCH.R`: ARMA-GARCH model implementation for volatility forecasting
- `EDA-SARIMA.R`: Exploratory data analysis and SARIMA modeling
- `train.csv`: Preprocessed training dataset
- `test.csv`: Preprocessed test dataset
- `lstm_model.keras`: Trained LSTM model
- `REPORT.pdf`: Detailed analysis and findings

## Data Sources

The project combines data from multiple sources:
- Daily cocoa prices from ICCO
- Weather data from Ghana
- Exchange rate data
- Consumer Price Index (CPI) data

## Models Implemented

1. **LSTM Neural Network** (`RNN.py`)
   - Implements a deep learning approach for time series forecasting
   - Features include deseasonalization, feature engineering, and prediction intervals
   - Includes hyperparameter tuning and model evaluation

2. **ARMA-GARCH Model** (`ARMA-GARCH.R`)
   - Models both the mean and volatility of cocoa prices
   - Suitable for capturing volatility clustering in financial time series

3. **SARIMA Model** (`EDA-SARIMA.R`)
   - Seasonal ARIMA modeling for time series forecasting
   - Includes comprehensive exploratory data analysis

## Usage

### Data Preprocessing
The data has already been preprocessed using `PREPROCESS.R`. The processed datasets are available as:
- `train.csv`: Training dataset
- `test.csv`: Test dataset

### Model Training and Evaluation
The LSTM model has already been trained and saved as `lstm_model.keras`. You can use the existing model for predictions or retrain it using the provided code.

## Dependencies

### Python Dependencies
```bash
pip install pandas numpy tensorflow scikit-learn statsmodels matplotlib joblib
```

### R Dependencies
```r
install.packages(c(
  "dplyr",        # Data manipulation
  "lubridate",    # Date handling
  "tidyr",        # Data tidying
  "readxl",       # Excel file reading
  "MASS",         # Statistical functions
  "tseries",      # Time series analysis
  "forecast",     # Forecasting functions
  "ggplot2",      # Data visualization
  "zoo",          # Time series objects
  "rugarch",      # GARCH modeling
  "astsa"         # Time series analysis
))
```

## Results

For detailed analysis, findings, and model performance metrics, please refer to `REPORT.pdf`. The report includes:

- Comprehensive model comparisons
- Performance metrics (RMSE, MAPE) for each approach
- Visualizations of forecasts and actual values
- Statistical analysis and significance tests
- Detailed discussion of model strengths and limitations

## Future Work

Potential improvements include:
- Integration of additional data sources
- Ensemble modeling combining multiple approaches
- Real-time forecasting capabilities
- Enhanced feature engineering
