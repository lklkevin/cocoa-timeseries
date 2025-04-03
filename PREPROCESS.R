library(dplyr)
library(lubridate)
library(tidyr)
library(readxl)

# load data
icco <- read.csv("Daily Prices_ICCO.csv", stringsAsFactors = FALSE)
ghana <- read.csv("Ghana_data.csv", stringsAsFactors = FALSE)
exchange <- read_excel("exchange.xlsx")
cpi <- read_excel("cpi.xlsx")

# convert dates
icco$Date <- dmy(icco$Date)
ghana$DATE <- ymd(ghana$DATE)
icco <- icco %>% arrange(Date)
ghana <- ghana %>% arrange(DATE)

# convert to numeric type
icco$ICCO.daily.price..US..tonne. <- as.numeric(gsub(",", "", icco$ICCO.daily.price..US..tonne.))
ghana$PRCP[ghana$PRCP == ""] <- 0
ghana$PRCP <- as.numeric(ghana$PRCP)
ghana$TAVG <- as.numeric(ghana$TAVG)
ghana$TMAX <- as.numeric(ghana$TMAX)
ghana$TMIN <- as.numeric(ghana$TMIN)

# add month column
ghana$MONTH <- floor_date(ghana$DATE, "month")
icco$MONTH <- floor_date(icco$Date, "month")

# monthly country average for Ghana
ghana_monthly_avg <- ghana %>%
  group_by(MONTH) %>%
  summarise(
    TAVG = mean(TAVG, na.rm = TRUE),
    TMAX = mean(TMAX, na.rm = TRUE),
    TMIN = mean(TMIN, na.rm = TRUE),
    PRCP = mean(PRCP, na.rm = TRUE)
  )

# monthly average cocoa price
icco_monthly_avg <- icco %>%
  group_by(MONTH) %>%
  summarise(
    ICCO_price = mean(ICCO.daily.price..US..tonne., na.rm = TRUE)
  )

# reshape function for world bank data
reshape_data <- function(df, values_to) {
  numeric_cols <- sapply(df, is.numeric)
  month_cols <- names(df)[numeric_cols]
  
  reshaped_data <- df %>%
    pivot_longer(
      cols = all_of(month_cols),  # Only numeric columns
      names_to = "month", 
      values_to = values_to
    ) %>%
    mutate(
      # Convert month column to proper date format
      MONTH = as.Date(paste0(
        substr(month, 1, 4), 
        "-", 
        substr(month, 6, 7), 
        "-01"
      ))
    ) %>%
    select(Country, MONTH, !!values_to)
  
  return(reshaped_data)
}

cpi <- reshape_data(cpi, "CPI")

adv_data <- cpi %>% filter(Country == "Advanced Economies") %>% select(MONTH, CPI)
world_data <- cpi %>% filter(Country == "World") %>% select(MONTH, CPI)
dev_data <- cpi %>% filter(Country == "Emerging Market and Developing Economies") %>% select(MONTH, CPI)

extract_exchange_rates <- function(exchange_data) {
  # NEER
  neer <- exchange_data %>% 
    filter(`Series Code` == "NEER") %>%
    reshape_data("NEER")
  
  # official exchange rate
  official <- exchange_data %>% 
    filter(`Series Code` == "DPANUSLCU") %>%
    reshape_data("official")
  
  # REER
  reer <- exchange_data %>% 
    filter(`Series Code` == "REER") %>%
    reshape_data("REER")
  
  combined_exchange_rates <- full_join(neer, official, by = c("Country", "MONTH")) %>%
    full_join(reer, by = c("Country", "MONTH"))
  
  return(combined_exchange_rates)
}

exchange[exchange == ".."] <- NA
exchange[, 5:ncol(exchange)] <- lapply(exchange[, 5:ncol(exchange)], as.numeric)
combined_exchange_rates <- extract_exchange_rates(exchange)

adv_data <- cpi %>% filter(Country == "Advanced Economies") %>% select(MONTH, CPI)
world_data <- cpi %>% filter(Country == "World") %>% select(MONTH, CPI)
dev_data <- cpi %>% filter(Country == "Emerging Market and Developing Economies") %>% select(MONTH, CPI)
ghana_data <- combined_exchange_rates %>% filter(Country == "Ghana") %>% select(-Country)
cote_ivoire_data <- combined_exchange_rates %>% filter(Country == "Cote d'Ivoire") %>% select(-Country)
nigeria_data <- combined_exchange_rates %>% filter(Country == "Nigeria") %>% select(-Country)
cameroon_data <- combined_exchange_rates %>% filter(Country == "Cameroon") %>% select(-Country)

combine_all_datasets <- function() {
  datasets <- list(
    adv_data = adv_data %>% rename(CPI_adv = CPI),
    world_data = world_data %>% rename(CPI_world = CPI),
    dev_data = dev_data %>% rename(CPI_dev = CPI),
    ghana_neer = ghana_data %>% 
      rename(Ghana_NEER = NEER) %>% 
      select(MONTH, Ghana_NEER),
    ghana_official = ghana_data %>% 
      rename(Ghana_official = official) %>% 
      select(MONTH, Ghana_official),
    ghana_reer = ghana_data %>% 
      rename(Ghana_REER = REER) %>% 
      select(MONTH, Ghana_REER),
    cote_ivoire_neer = cote_ivoire_data %>% 
      rename(CoteIvoire_NEER = NEER) %>% 
      select(MONTH, CoteIvoire_NEER),
    cote_ivoire_official = cote_ivoire_data %>% 
      rename(CoteIvoire_official = official) %>% 
      select(MONTH, CoteIvoire_official),
    cote_ivoire_reer = cote_ivoire_data %>% 
      rename(CoteIvoire_REER = REER) %>% 
      select(MONTH, CoteIvoire_REER),
    nigeria_neer = nigeria_data %>% 
      rename(Nigeria_NEER = NEER) %>% 
      select(MONTH, Nigeria_NEER),
    nigeria_official = nigeria_data %>% 
      rename(Nigeria_official = official) %>% 
      select(MONTH, Nigeria_official),
    nigeria_reer = nigeria_data %>% 
      rename(Nigeria_REER = REER) %>% 
      select(MONTH, Nigeria_REER),
    cameroon_neer = cameroon_data %>% 
      rename(Cameroon_NEER = NEER) %>% 
      select(MONTH, Cameroon_NEER),
    cameroon_official = cameroon_data %>% 
      rename(Cameroon_official = official) %>% 
      select(MONTH, Cameroon_official),
    cameroon_reer = cameroon_data %>% 
      rename(Cameroon_REER = REER) %>% 
      select(MONTH, Cameroon_REER),
    ghana_monthly_avg = ghana_monthly_avg,
    icco_monthly_avg = icco_monthly_avg
  )
  
  # reduce with full_join to keep all dates and NAs
  combined_data <- Reduce(
    function(x, y) full_join(x, y, by = "MONTH"), 
    datasets
  )
  
  return(combined_data)
}

comprehensive_data <- combine_all_datasets()
# only include months with ICCO data
comprehensive_data <- comprehensive_data %>%
  mutate(across(where(is.numeric), ~replace(., is.nan(.), NA))) %>%
  filter(MONTH >= as.Date("1994-10-01")) %>% arrange(MONTH)

# split data into train and test sets
test_start_date <- comprehensive_data$MONTH[nrow(comprehensive_data) - 4]
train_data <- comprehensive_data %>% filter(MONTH < test_start_date)
test_data <- comprehensive_data %>% filter(MONTH >= test_start_date)

write.csv(train_data,"train.csv", row.names = FALSE)
write.csv(test_data,"test.csv", row.names = FALSE)
