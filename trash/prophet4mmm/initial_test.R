# pruebas tutorial prophet facebook
# https://facebook.github.io/prophet/docs/quick_start.html#r-api

library(prophet)

# quick start
# ... asume que el proyecto está bajado desde gihub
df <- read.csv('../../github/prophet/examples/example_wp_log_peyton_manning.csv')
summary(df)
tail(df)

m <- prophet(df)
future <- make_future_dataframe(m, periods = 365)
tail(future)

forecast <- predict(m, future)
tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])

plot(m, forecast)

prophet_plot_components(m, forecast)




# saturating forecasts
df <- read.csv('~/../Downloads/prophet-master/examples/example_wp_log_R.csv')
df$cap <- 8.5
m <- prophet(df, growth = 'logistic')

future <- make_future_dataframe(m, periods = 1826)
future$cap <- 8.5
fcst <- predict(m, future)
plot(m, fcst)

df$y <- 10 - df$y
df$cap <- 6
df$floor <- 1.5
future$cap <- 6
future$floor <- 1.5
m <- prophet(df, growth = 'logistic')
fcst <- predict(m, future)
plot(m, fcst)





# trend changepoints (from quick start)
plot(m, forecast) + add_changepoints_to_plot(m)

m <- prophet(df, changepoint.prior.scale = 0.5)
forecast <- predict(m, future)
plot(m, forecast) + add_changepoints_to_plot(m)

m <- prophet(df, changepoint.prior.scale = 0.01)
forecast <- predict(m, future)
plot(m, forecast) + add_changepoints_to_plot(m)

m <- prophet(df, changepoints = c('2014-01-01'))
forecast <- predict(m, future)
plot(m, forecast) + add_changepoints_to_plot(m)





# seasonality & regressors (from quickstart)
library(dplyr)
playoffs <- data_frame(
  holiday = 'playoff',
  ds = as.Date(c('2008-01-13', '2009-01-03', '2010-01-16',
                 '2010-01-24', '2010-02-07', '2011-01-08',
                 '2013-01-12', '2014-01-12', '2014-01-19',
                 '2014-02-02', '2015-01-11', '2016-01-17',
                 '2016-01-24', '2016-02-07')),
  lower_window = 0,
  upper_window = 1
)
superbowls <- data_frame(
  holiday = 'superbowl',
  ds = as.Date(c('2010-02-07', '2014-02-02', '2016-02-07')),
  lower_window = 0,
  upper_window = 1
)
holidays <- bind_rows(playoffs, superbowls)

m <- prophet(df, holidays = holidays)
forecast <- predict(m, future)

forecast %>% 
  select(ds, playoff, superbowl) %>% 
  filter(abs(playoff + superbowl) > 0) %>%
  tail(10)

prophet_plot_components(m, forecast)

m <- prophet(holidays = holidays)
m <- add_country_holidays(m, country_name = 'US')
m <- fit.prophet(m, df)

m$train.holiday.names

forecast <- predict(m, future)
prophet_plot_components(m, forecast)

m <- prophet(df)
prophet:::plot_yearly(m)

m <- prophet(df, yearly.seasonality = 20)
prophet:::plot_yearly(m)


m <- prophet(weekly.seasonality=FALSE)
m <- add_seasonality(m, name='monthly', period=30.5, fourier.order=5)
m <- fit.prophet(m, df)
forecast <- predict(m, future)
prophet_plot_components(m, forecast)


is_nfl_season <- function(ds) {
  dates <- as.Date(ds)
  month <- as.numeric(format(dates, '%m'))
  return(month > 8 | month < 2)
}
df$on_season <- is_nfl_season(df$ds)
df$off_season <- !is_nfl_season(df$ds)

m <- prophet(weekly.seasonality=FALSE)
m <- add_seasonality(m, name='weekly_on_season', period=7, fourier.order=3, condition.name='on_season')
m <- add_seasonality(m, name='weekly_off_season', period=7, fourier.order=3, condition.name='off_season')
m <- fit.prophet(m, df)

future$on_season <- is_nfl_season(future$ds)
future$off_season <- !is_nfl_season(future$ds)
forecast <- predict(m, future)
prophet_plot_components(m, forecast)



nfl_sunday <- function(ds) {
  dates <- as.Date(ds)
  month <- as.numeric(format(dates, '%m'))
  as.numeric((weekdays(dates) == "Sunday") & (month > 8 | month < 2))
}
df$nfl_sunday <- nfl_sunday(df$ds)

m <- prophet()
m <- add_regressor(m, 'nfl_sunday')
m <- fit.prophet(m, df)

future$nfl_sunday <- nfl_sunday(future$ds)

forecast <- predict(m, future)
prophet_plot_components(m, forecast)

