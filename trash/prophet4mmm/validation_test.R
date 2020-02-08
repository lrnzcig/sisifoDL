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
forecast <- predict(m, future)
tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])

plot(m, forecast)

prophet_plot_components(m, forecast)



# uncertainty
m <- prophet(df, mcmc.samples = 300)
forecast <- predict(m, future)
prophet_plot_components(m, forecast)



# cross-validation
df.cv <- cross_validation(m, initial = 730, period = 180, horizon = 365, units = 'days')
head(df.cv)
plot_cross_validation_metric(df.cv, metric = 'mape')

df.p <- performance_metrics(df.cv)
head(df.p)
