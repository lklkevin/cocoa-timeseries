# Load necessary libraries
library(readr)
library(dplyr)
library(lubridate)
library(MASS)
library(tseries)
library(forecast)
library(ggplot2)
library(zoo)
library(rugarch)

# ----------------------------
# 1. Data Loading and Preparation
# ----------------------------

# Load the datasets
data_train <- read_csv("train.csv")
data_test <- read_csv("test.csv")

# Clean test data by removing rows with all NAs
data_test <- data_test %>%
  filter_all(any_vars(!is.na(.)))

# Standardize column names to uppercase
data_train <- rename_with(data_train, toupper)
data_test <- rename_with(data_test, toupper)

# Convert month to Date format
data_train$MONTH <- as.Date(paste0(data_train$MONTH, "-01"))
data_test$MONTH <- as.Date(paste0(data_test$MONTH, "-01"))

# Convert to tibble for better compatibility
data_train <- as_tibble(data_train)
data_test <- as_tibble(data_test)

# Filter training data to only include data past 2020
data_train <- data_train %>% 
  filter(MONTH > as.Date("2020-01-01"))

# Handle missing values using interpolation
data_train <- data_train %>% mutate(across(where(is.numeric), ~na.approx(., na.rm = FALSE)))
data_test <- data_test %>% mutate(across(where(is.numeric), ~na.locf(., na.rm = FALSE)))

# ----------------------------
# 2. Feature Engineering
# ----------------------------

# Create lagged versions of relevant predictors 
data_train <- data_train %>%
  mutate(across(c('GHANA_OFFICIAL', 'COTEIVOIRE_OFFICIAL', 'NIGERIA_OFFICIAL', 'CAMEROON_OFFICIAL', 
                  'TAVG', 'TMAX', 'TMIN', 'CPI_ADV', 'CPI_WORLD', 'CPI_DEV'),
                ~lag(.), .names = "{.col}_lag1"))

data_test <- data_test %>%
  mutate(across(c('GHANA_OFFICIAL', 'COTEIVOIRE_OFFICIAL', 'NIGERIA_OFFICIAL', 'CAMEROON_OFFICIAL', 
                  'TAVG', 'TMAX', 'TMIN', 'CPI_ADV', 'CPI_WORLD', 'CPI_DEV'),
                ~lag(.), .names = "{.col}_lag1"))

# Create squared and cubic transformations of predictors
transform_vars <- c('GHANA_OFFICIAL', 'COTEIVOIRE_OFFICIAL', 'NIGERIA_OFFICIAL', 'CAMEROON_OFFICIAL',
                    'TAVG', 'TMAX', 'TMIN', 'CPI_ADV', 'CPI_WORLD', 'CPI_DEV')

# Apply transformations to training data
data_train <- data_train %>%
  mutate(across(all_of(transform_vars), list(sq = ~.^2, cub = ~.^3), .names = "{.col}_{.fn}"))

# Apply transformations to testing data
data_test <- data_test %>%
  mutate(across(all_of(transform_vars), list(sq = ~.^2, cub = ~.^3), .names = "{.col}_{.fn}"))

# ----------------------------
# 3. Regression Modeling
# ----------------------------

# Select relevant columns for regression
train_data_model <- data_train %>%
  dplyr::select(ICCO_PRICE, 
                GHANA_OFFICIAL, COTEIVOIRE_OFFICIAL, NIGERIA_OFFICIAL, CAMEROON_OFFICIAL,
                TAVG, TMAX, TMIN, CPI_ADV, CPI_WORLD, CPI_DEV,
                starts_with("GHANA_OFFICIAL_lag1"), starts_with("COTEIVOIRE_OFFICIAL_lag1"),
                starts_with("NIGERIA_OFFICIAL_lag1"), starts_with("CAMEROON_OFFICIAL_lag1"),
                starts_with("TAVG_lag1"), starts_with("TMAX_lag1"), starts_with("TMIN_lag1"),
                starts_with("CPI_ADV_lag1"), starts_with("CPI_WORLD_lag1"), starts_with("CPI_DEV_lag1"),
                starts_with("GHANA_OFFICIAL_sq"), starts_with("COTEIVOIRE_OFFICIAL_sq"), 
                starts_with("NIGERIA_OFFICIAL_sq"), starts_with("CAMEROON_OFFICIAL_sq"),
                starts_with("TAVG_sq"), starts_with("TMAX_sq"), starts_with("TMIN_sq"),
                starts_with("CPI_ADV_sq"), starts_with("CPI_WORLD_sq"), starts_with("CPI_DEV_sq"),
                starts_with("GHANA_OFFICIAL_cub"), starts_with("COTEIVOIRE_OFFICIAL_cub"), 
                starts_with("NIGERIA_OFFICIAL_cub"), starts_with("CAMEROON_OFFICIAL_cub"),
                starts_with("TAVG_cub"), starts_with("TMAX_cub"), starts_with("TMIN_cub"),
                starts_with("CPI_ADV_cub"), starts_with("CPI_WORLD_cub"), starts_with("CPI_DEV_cub"))

# Define and fit full regression model
full_model <- lm(ICCO_PRICE ~ ., data = train_data_model)

# Perform backward selection based on AIC
stepwise_model <- stepAIC(full_model, direction = "backward")

# Print model summary
cat("\nRegression Model Summary:\n")
print(summary(stepwise_model))
cat("Final AIC value:", AIC(stepwise_model), "\n")

# ----------------------------
# 4. Residual Analysis (ARMA)
# ----------------------------

# Extract residuals and fit ARMA model
residuals_ts <- ts(residuals(stepwise_model), 
                   start = c(year(min(data_train$MONTH)), month(min(data_train$MONTH))), 
                   frequency = 12)

arma_fit <- Arima(residuals_ts, c(1,0,0))

cat("\nARMA Model Summary:\n")
print(summary(arma_fit))
cat("\nARMA Residual Diagnostics:\n")
print(checkresiduals(arma_fit, plot = FALSE))

# ----------------------------
# 5. GARCH Modeling
# ----------------------------

# Get pure residuals (data - regression - arma)
pure_residuals <- residuals_ts - fitted(arma_fit)

# Take log difference of pure residuals for GARCH modeling
log_diff_residuals <- diff(log(abs(pure_residuals) + 1))  # Adding 1 to avoid log(0)

# Fit GARCH model to the differenced log residuals
garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
)

garch_model <- ugarchfit(spec = garch_spec, data = log_diff_residuals)

cat("\nGARCH Model Summary:\n")
print(garch_model)

# Forecast GARCH components
garch_forecast <- ugarchforecast(garch_model, n.ahead = nrow(data_test))

# Get forecasted log differences
forecast_log_diff <- garch_forecast@forecast$seriesFor

# Transform back to original scale
if(length(forecast_log_diff) > 0) {
  last_log_value <- log(tail(abs(pure_residuals), 1) + 1)
  forecast_log_values <- cumsum(c(last_log_value, forecast_log_diff))[-1]
  forecast_garch_residuals <- exp(forecast_log_values) - 1
} else {
  forecast_garch_residuals <- rep(0, nrow(data_test))
}

# ----------------------------
# 6. Forecasting
# ----------------------------

# Prepare test data by adding last training row
last_train_row <- data_train %>% 
  slice(n()) %>% 
  dplyr::select(GHANA_OFFICIAL, COTEIVOIRE_OFFICIAL, NIGERIA_OFFICIAL, CAMEROON_OFFICIAL,
                TAVG, TMAX, TMIN, CPI_ADV, CPI_WORLD, CPI_DEV)

data_test <- bind_rows(last_train_row, data_test)

# Create lagged variables for test data
data_test <- data_test %>%
  mutate(across(c('GHANA_OFFICIAL', 'COTEIVOIRE_OFFICIAL', 'NIGERIA_OFFICIAL', 'CAMEROON_OFFICIAL', 
                  'TAVG', 'TMAX', 'TMIN', 'CPI_ADV', 'CPI_WORLD', 'CPI_DEV'),
                ~lag(.), .names = "{.col}_lag1"))

# Prepare test data for prediction
test_data_model <- data_test %>%
  dplyr::select(ICCO_PRICE, 
                GHANA_OFFICIAL, COTEIVOIRE_OFFICIAL, NIGERIA_OFFICIAL, CAMEROON_OFFICIAL,
                TAVG, TMAX, TMIN, CPI_ADV, CPI_WORLD, CPI_DEV, 
                starts_with("GHANA_OFFICIAL_lag1"), starts_with("COTEIVOIRE_OFFICIAL_lag1"),
                starts_with("NIGERIA_OFFICIAL_lag1"), starts_with("CAMEROON_OFFICIAL_lag1"),
                starts_with("TAVG_lag1"), starts_with("TMAX_lag1"), starts_with("TMIN_lag1"),
                starts_with("CPI_ADV_lag1"), starts_with("CPI_WORLD_lag1"), starts_with("CPI_DEV_lag1"),
                starts_with("GHANA_OFFICIAL_sq"), starts_with("COTEIVOIRE_OFFICIAL_sq"), 
                starts_with("NIGERIA_OFFICIAL_sq"), starts_with("CAMEROON_OFFICIAL_sq"),
                starts_with("TAVG_sq"), starts_with("TMAX_sq"), starts_with("TMIN_sq"),
                starts_with("CPI_ADV_sq"), starts_with("CPI_WORLD_sq"), starts_with("CPI_DEV_sq"),
                starts_with("GHANA_OFFICIAL_cub"), starts_with("COTEIVOIRE_OFFICIAL_cub"), 
                starts_with("NIGERIA_OFFICIAL_cub"), starts_with("CAMEROON_OFFICIAL_cub"),
                starts_with("TAVG_cub"), starts_with("TMAX_cub"), starts_with("TMIN_cub"),
                starts_with("CPI_ADV_cub"), starts_with("CPI_WORLD_cub"), starts_with("CPI_DEV_cub"))

# Generate forecasts from each component
regression_forecast <- predict(stepwise_model, newdata = test_data_model, interval = "prediction")
residual_forecast <- forecast(arma_fit, h = nrow(data_test))

# Get GARCH standard deviations (not variance)
garch_sd <- sigma(garch_forecast)  # This gives standard deviation directly

# Combine all components for final forecast
final_forecast <- regression_forecast[, 1][-1] + 
  residual_forecast$mean[-1] + 
  forecast_garch_residuals

# Calculate proper confidence intervals
# Use 1.96 for 95% CI (z-score for normal distribution)
confidence_width <- 1.96 * sqrt(
  # Regression prediction variance
  (regression_forecast[,1] - regression_forecast[,2])[2:nrow(data_test)]^2 +
    # ARMA forecast variance
    (residual_forecast$upper[,2] - residual_forecast$mean)[1:(nrow(data_test)-1)]^2 +
    # GARCH variance
    garch_sd^2
)

# Create forecast dataframe with properly aligned CIs
forecast_df <- tibble::tibble(
  Month = data_test$MONTH[-1],
  Actual = data_test$ICCO_PRICE[-1],
  Forecast = final_forecast,
  Lower = final_forecast - confidence_width,
  Upper = final_forecast + confidence_width
) %>% 
  na.omit()
# ----------------------------
# 7. Evaluation and Visualization
# ----------------------------

# Calculate performance metrics
rmse <- sqrt(mean((forecast_df$Actual - forecast_df$Forecast)^2, na.rm = TRUE))
mape <- mean(abs((forecast_df$Actual - forecast_df$Forecast) / forecast_df$Actual), na.rm = TRUE) * 100

cat("\nModel Performance:\n")
cat("RMSE:", rmse, "\n")
cat("MAPE:", mape, "%\n")

# Create visualization
p <- ggplot() +
  geom_line(data = data_train, aes(x = MONTH, y = ICCO_PRICE, color = "Training Data"), size = 1) +
  geom_line(data = forecast_df, aes(x = Month, y = Actual, color = "Actual (Test)"), size = 1, linetype = "dashed") +
  geom_line(data = forecast_df, aes(x = Month, y = Forecast, color = "Forecast"), size = 1) +
  geom_ribbon(data = forecast_df, aes(x = Month, ymin = Lower, ymax = Upper, fill = "95% CI"), alpha = 0.3) +
  geom_point(data = forecast_df, aes(x = Month, y = Actual, color = "Actual (Test)"), size = 2) +
  geom_point(data = forecast_df, aes(x = Month, y = Forecast, color = "Forecast"), size = 2) +
  scale_color_manual(
    name = "Lines",
    values = c("Training Data" = "blue", "Actual (Test)" = "blue", "Forecast" = "red"),
    breaks = c("Training Data", "Actual (Test)", "Forecast")
  ) +
  scale_fill_manual(name = "Interval", values = c("95% CI" = "gray")) +
  labs(
    title = "Cocoa Price Forecast with Regression + ARMA + GARCH Components",
    subtitle = paste("RMSE:", round(rmse, 2), "| MAPE:", round(mape, 2), "%"),
    x = "Month", 
    y = "ICCO Price"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(color = "gray40")
  ) +
  xlim(as.Date("2020-01-01"), as.Date("2025-04-01"))

# Display the plot
print(p)
