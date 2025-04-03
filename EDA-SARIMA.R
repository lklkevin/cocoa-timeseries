##############################################################################
# 1. SETUP
##############################################################################

# Clear environment
rm(list = ls())

# Load necessary libraries
library(dplyr)
library(lubridate)
library(ggplot2)
library(forecast)
library(tseries)
library(astsa)
install.packages("dplyr")

##############################################################################
# 2. READ CSV & CREATE FULL MONTHLY SEQUENCE
##############################################################################

# Read the CSV from the local path (as provided by the user)
df <- read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)

# Convert 'MONTH' column to a Date (assuming YYYY-MM-DD format)
df$MONTH <- as.Date(df$MONTH)

# Sort by MONTH (good practice for time series)
df <- df %>% arrange(MONTH)

# Define start/end for monthly series
start_date <- as.Date("1994-10-01")
end_date   <- as.Date("2024-10-01")

# Create a complete sequence of monthly dates
all_months <- data.frame(
  MONTH = seq.Date(from = start_date, to = end_date, by = "month")
)

# Left join your original data onto this full monthly grid
# => ensures no missing month-rows from 1994-10 to 2024-10
df_full <- all_months %>%
  left_join(df, by = "MONTH")

##############################################################################
# 3. INITIAL EXPLORATION ON df_full
##############################################################################

# Inspect the structure
str(df_full)

# Quick summary
summary(df_full)

# List the specific variables of interest
vars <- c("ICCO_price", "CPI_world", "TAVG", "PRCP", "Ghana_official")

# Calculate NA% for each variable
na_pct <- sapply(vars, function(var) {
  mean(is.na(df_full[[var]])) * 100
})
na_pct

# Check missing values by column
cat("Missing values per column:\n")
sapply(df_full, function(x) sum(is.na(x)))

# Basic line plots to see the raw data
ggplot(df_full, aes(x = MONTH, y = ICCO_price)) +
  geom_line() +
  labs(title = "Time Series of Cocoa Price", x = "Month", y = "Price")

ggplot(df_full, aes(x = MONTH, y = TAVG)) +
  geom_line() +
  labs(title = "Time Series of Monthly Avg Temperature (TAVG)", x = "Month", y = "Temperature")

ggplot(df_full, aes(x = MONTH, y = PRCP)) +
  geom_line() +
  labs(title = "Time Series of Monthly Precipitation (PRCP)", x = "Month", y = "Precipitation")

##############################################################################
# 4. MONTHLY-CLIMATOLOGY IMPUTATION (for TAVG & PRCP)
##############################################################################
# We create a new data frame "df_imputed".
# For each missing TAVG/PRCP value, fill it with the mean TAVG/PRCP for that MONTH (across all years).

df_with_month_num <- df_full %>%
  mutate(month_num = month(MONTH))  # 1 = January, 12 = December

# Calculate monthly means for TAVG & PRCP (excluding NA)
monthly_means <- df_with_month_num %>%
  group_by(month_num) %>%
  summarize(
    mean_temp   = mean(TAVG, na.rm = TRUE),
    mean_precip = mean(PRCP, na.rm = TRUE)
  )

# Impute TAVG & PRCP in a new data frame
df_imputed <- df_with_month_num %>%
  left_join(monthly_means, by = "month_num") %>%
  mutate(
    TAVG_imputed = ifelse(is.na(TAVG), mean_temp, TAVG),
    PRCP_imputed = ifelse(is.na(PRCP), mean_precip, PRCP)
  ) %>%
  select(-month_num, -mean_temp, -mean_precip)

# Verify no more missing values in the new columns
cat("NAs in df_imputed$TAVG_imputed:", sum(is.na(df_imputed$TAVG_imputed)), "\n")
cat("NAs in df_imputed$PRCP_imputed:", sum(is.na(df_imputed$PRCP_imputed)), "\n")

# Write out if desired
write.csv(df_imputed, "train_imputed.csv", row.names = FALSE)

##############################################################################
# 5. CREATE TIME SERIES OBJECTS
##############################################################################
# We have monthly data from 1994-10 to 2024-10 => 361 data points total
# So set start/end = c(1994,10) & c(2024,10), frequency = 12

ts_price <- ts(
  df_imputed$ICCO_price,
  start     = c(1994, 10),
  end       = c(2024, 10),
  frequency = 12
)

ts_temp <- ts(
  df_imputed$TAVG_imputed,
  start     = c(1994, 10),
  end       = c(2024, 10),
  frequency = 12
)

ts_precip <- ts(
  df_imputed$PRCP_imputed,
  start     = c(1994, 10),
  end       = c(2024, 10),
  frequency = 12
)

ts_ghanaex <- ts(
  df_imputed$Ghana_official,
  start     = c(1994, 10),
  end       = c(2024, 10),
  frequency = 12
)

ts_cpi <- ts(
  df_imputed$CPI_world,
  start     = c(1994, 10),
  end       = c(2024, 10),
  frequency = 12
)
ts_nigeriaex <- ts(
  df_imputed$Nigeria_official,
  start     = c(1994, 10),
  end       = c(2024, 10),
  frequency = 12
)

# Plot the series
par(
  cex      = 1.2,   # overall text scaling
  cex.main = 1.4,   # title text size
  cex.lab  = 1.3,   # x- and y-axis label size
  cex.axis = 1.1
)

plot(ts_price, main = "Time-Series Plot of Cocoa Futures Prices",
     xlab = "Time", ylab = "Cocoa Futures Price (USD)")

plot(ts_temp, main = "Ghana Monthly Avg. Temperature (Fahrenheit)",
     xlab = "Time", ylab = "Avg Monthly Temperature (F)")

plot(ts_precip, main = "Ghana Monthly Precipitation",
     xlab = "Time", ylab = "Precipitation")

plot(ts_ghanaex, main = "Time-Series Plot of Ghana/USD Exchange Rate",
     xlab = "Time", ylab = "Exchange Rate (Cedi/USD)")

plot(ts_cpi, main = "Time-Series Plot of Global CPI",
     xlab = "Time", ylab = "Global CPI")

##############################################################################
# 6. DECOMPOSITION (STL)
##############################################################################
stl_price <- stl(ts_price, s.window = "periodic")
plot(stl_price, main = "STL Decomposition of Cocoa Price")

stl_precip <- stl(ts_precip, s.window = "periodic")
plot(stl_precip, main = "STL Decomposition of Monthly Precipitation")

stl_temp <- stl(ts_temp, s.window = "periodic")
plot(stl_temp, main = "STL Decomposition of Monthly Avg Temperature")

stl_ghanaex <- stl(ts_ghanaex, s.window = "periodic")
plot(stl_ghanaex, main = "STL Decomposition of Ghana/USD Exchange Rate")


##############################################################################
# 7. STATIONARITY CHECK & AUTOCORRELATION
##############################################################################
acf(ts_price) 
pacf(ts_price)
adf_test_price <- adf.test(ts_price)
adf_test_price 
ndiffs(ts_price) 

acf(ts_precip)
pacf(ts_precip)
adf_test_precip <- adf.test(ts_precip)
adf_test_precip 
ndiffs(ts_precip)

acf(ts_temp, main = "ACF of TAVG Time-Series", xlab = "Lag (Years)")
pacf(ts_temp)
adf_test_tavg <- adf.test(ts_temp)
adf_test_tavg
ndiffs(ts_temp)

acf(ts_ghanaex)
pacf(ts_ghanaex)
adf_test_ghanaex <- adf.test(ts_ghanaex)
adf_test_ghanaex
ndiffs(ts_ghanaex)

# Preliminary ARIMA fits
auto.arima(ts_price, seasonal = TRUE)  
auto.arima(ts_precip, seasonal = TRUE) 
auto.arima(ts_temp, seasonal = TRUE)   

##############################################################################
# 8. DATA DIFFERENCING
##############################################################################
price_diff  <- diff(ts_price,   differences = 1)
precip_diff <- diff(ts_precip, differences = 1)

plot(price_diff, main = "Plot of Differenced Cocoa Price Time-Series",
     ylab = "Price (1st Diff)", xlab = "Time")

plot(precip_diff, main = "Differenced Precipitation",
     ylab = "Precip (1st Diff)", xlab = "Time")

par(mfrow = c(1,1))
acf(price_diff,   main = "ACF: Differenced Price")
pacf(price_diff,  main = "PACF: Differenced Price")
acf(precip_diff,  main = "ACF: Differenced Precip")
Pacf(precip_diff, main = "PACF: Differenced Precip")

auto.arima(price_diff,  seasonal = TRUE)
auto.arima(precip_diff, seasonal = TRUE)

##############################################################################
# 9. CROSS-CORRELATION ANALYSIS
##############################################################################
ccf(ts_price,   ts_temp,   lag.max = 12,
    main = "Cross-Correlation: Price vs. TAVG")

ccf(ts_price,   ts_precip, lag.max = 12,
    main = "Cross-Correlation: Price vs. PRCP")

ccf(price_diff, ts_precip, lag.max = 12,
    main = "Cross-Corr: 1st-Diff Price vs. PRCP")

ccf(price_diff, ts_temp,   lag.max = 12,
    main = "Cross-Corr: 1st-Diff Price vs. TAVG")
month_number <- month(df_imputed$MONTH)
# 2) Convert this numeric vector into a time series with the same start/end & frequency
ts_month_num <- ts(
  month_number,
  start     = c(1994, 10),   # same as ts_price
  end       = c(2024, 10),   # same as ts_price
  frequency = 12
)

##############################################################################
# B. PLOT THE CROSS-CORRELATION FUNCTION
##############################################################################

ccf(ts_price, ts_month_num, 
    lag.max = 12,
    main = "Cross-Correlation: Price vs. Month Number")

ts_price_2022 <- window(ts_price, start = c(2022, 1))

# 2) Print (or return) this new time series
ts_price_2022
sarima(ts_price_2022, 4,1,4,1,1,1,12)
sarima.for(ts_price_2022, 4, 4,1,4,1,1,1,12)
official_vals <- c(7930.125238, 10353.04714, 10709.30545, 9827.304211)
ts_price_2020 <- window(ts_price, start = c(2020,1))
sarima(ts_price_2020, 4,1,4,1,1,1,12)
# We need to place these values at the correct time indices,
# which is the next 4 months after the end of ts_price_2022.

# 3a) Determine the start (year, month) of the forecast horizon
end_time <- end(ts_price_2022)            # e.g. c(YYYY, MM)
start_forecast <- c(end_time[1], end_time[2] + 1)

# If the month value exceeds 12, adjust year/month accordingly
if (start_forecast[2] > 12) {
  start_forecast[1] <- start_forecast[1] + 1
  start_forecast[2] <- start_forecast[2] - 12
}

# 3b) Create a small time series for the 4 official values
official_ts <- ts(official_vals, start = start_forecast, frequency = 12)

# 3c) Overlay on the existing forecast plot
# 'sarima.for()' already made a plot, so we can just add lines/points:
lines(official_ts, col = "red", lwd = 2)
points(official_ts, col = "red", pch = 16)
# Store the result of sarima.for() in an object
sarima_fit <- sarima.for(ts_price_2022, 
                         n.ahead = 4, 
                         p = 4, d = 1, q = 4, 
                         P = 1, D = 1, Q = 1, 
                         S = 12)

# Extract the forecasts (this is the predicted mean for each forecast horizon)
model_forecasts <- sarima_fit$pred

# 1) RMSE
rmse <- sqrt(mean((official_vals - model_forecasts)^2))

# 2) MAPE
mape <- mean(abs((official_vals - model_forecasts) / official_vals)) * 100

# Print out RMSE and MAPE
cat("RMSE =", rmse, "\n")
cat("MAPE =", mape, "%\n")
##############################################################################
# 10. CREATE A TIME SERIES FOR 2022 ONWARD & SARIMAX WITH MONTH
##############################################################################

# Let's say we want to model the data from 2022-01 onward
ts_price_2022 <- window(ts_price, start = c(2022, 1))
end(ts_price_2022)

### ADDED CODE: CREATE A NUMERIC-MONTH TIME SERIES (1..12) AS EXOGENOUS ###
# We'll map each Date in df_imputed$MONTH to its numeric month.
# Then convert that to a time series with the same start/end as ts_price.
ts_month_num <- ts(
  data       = month(df_imputed$MONTH),    # numeric month (1..12)
  start      = c(1994, 10),
  end        = c(2024, 10),
  frequency  = 12
)

# Window out the same range for the exogenous variable (2022 onward)
month_2022 <- window(ts_month_num, start = c(2022, 1))

# Example: Fit SARIMAX(4,1,4)(1,1,1)[12] using 'month_2022' as xreg
sarimax_fit <- sarima(
  x    = ts_price_2022,
  p    = 2,
  d    = 1,
  q    = 2,
  P    = 1,
  D    = 1,
  Q    = 1,
  S    = 12,
  xreg = month_2022
)
# This prints out the model summary in the console.
##############################################################################
# 11. FORECAST WITH SARIMAX (MONTH AS EXOGENOUS) - NEXT 4 MONTHS
##############################################################################

# We have official values for the next 4 months' PRICE (this is to evaluate forecast):
official_vals <- c(7930.125238, 10353.04714, 10709.30545, 9827.304211)

# Identify the next 4 months after the end of ts_price_2022
end_time <- end(ts_price_2022)   # e.g. c(2024, 10)
start_forecast <- c(end_time[1], end_time[2] + 1)
if (start_forecast[2] > 12) {
  start_forecast[1] <- start_forecast[1] + 1
  start_forecast[2] <- start_forecast[2] - 12
}

# For these 4 months, the numeric "month" values are:
# If the last in-sample month is October (10),
# then the next 4 months are: 11 (Nov), 12 (Dec), 1 (Jan), 2 (Feb).
new_months <- c(11, 12, 1, 2)

m <- sarima(
  x        = ts_price_2022,
  n.ahead  = 4,
  p        = 3,
  d        = 1,
  q        = 1,
  P        = 0,
  D        = 1,
  Q        = 1,
  S        = 12,
  xreg     = month_2022,    # in-sample exogenous
  newxreg  = new_months     # next 4 months exogenous
)
summary(m)
auto.arima(ts_price)
# Forecast with month as exogenous
sarimax_for <- sarima.for(
  x        = ts_price_2022,
  n.ahead  = 4,
  p        = 3,
  d        = 1,
  q        = 2,
  P        = 1,
  D        = 1,
  Q        = 1,
  S        = 12,
  xreg     = month_2022,    # in-sample exogenous
  newxreg  = new_months     # next 4 months exogenous
)

# Extract the forecasts from the model
model_forecasts <- sarimax_for$pred

# Evaluate the forecast against official values
rmse <- sqrt(mean((official_vals - model_forecasts)^2))
mape <- mean(abs((official_vals - model_forecasts) / official_vals)) * 100

cat("RMSE =", rmse, "\n")
cat("MAPE =", mape, "%\n")

# Overlay the official values on the forecast plot
official_ts <- ts(official_vals, start = start_forecast, frequency = 12)
lines(official_ts, col = "red", lwd = 2)
points(official_ts, col = "red", pch = 16)
