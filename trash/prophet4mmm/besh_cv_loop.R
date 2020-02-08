# BLUCLES CV PARA SELECCION DE PRIORS

# 0. test de la función CV
prophet.cv.loop(train %>%
                  select(ds=Fecha, y=ventas_online_uds,
                         otros_precio_medio, trafico_web_sesiones_totales),
                initial=106*7,
                period=14,
                horizon=28,
                units="days",
                model.function=function(data_train) {
                  m_ser <- prophet(yearly.seasonality=10,
                                   weekly.seasonality=FALSE,
                                   daily.seasonality=FALSE,
                                   seasonality.prior.scale=10, #10
                                   changepoint.prior.scale=0.05) #0.05
                  m_ser <- add_seasonality(m_ser, name='monthly', period=30.5,
                                           fourier.order=8, prior.scale=10) #10
                  m_ser <- add_regressor(m_ser, 'otros_precio_medio', prior.scale=5) #5
                  m_ser <- add_regressor(m_ser, 'trafico_web_sesiones_totales', prior.scale=5) #5
                  m_ser <- fit.prophet(m_ser, 
                                       train %>%
                                         select(ds=Fecha, y=ventas_online_uds,
                                                otros_precio_medio, trafico_web_sesiones_totales))
                  return(m_ser)
                })


# 1. CV grid piors 2 variables explicativas
res_df1 <- c()
for (prior_scale_1 in c(0.01, 0.1, 1, 5, 10)) {
  for (prior_scale_2 in c(0.01, 0.1, 1, 5, 10)) {
    res <- prophet.cv.loop(train %>%
                             select(ds=Fecha, y=ventas_online_uds,
                                    otros_precio_medio, trafico_web_sesiones_totales),
                           initial=106*7,
                           period=14,
                           horizon=28,
                           units="days",
                           model.function=function(data_train) {
                             m_ser <- prophet(yearly.seasonality=10,
                                              weekly.seasonality=FALSE,
                                              daily.seasonality=FALSE,
                                              seasonality.prior.scale=10, #10
                                              changepoint.prior.scale=0.05) #0.05
                             m_ser <- add_seasonality(m_ser, name='monthly', period=30.5,
                                                      fourier.order=8, prior.scale=10) #10
                             m_ser <- add_regressor(m_ser, 'otros_precio_medio', 
                                                    prior.scale=prior_scale_1) #5
                             m_ser <- add_regressor(m_ser, 'trafico_web_sesiones_totales', 
                                                    prior.scale=prior_scale_2) #5
                             m_ser <- fit.prophet(m_ser, 
                                                  train %>%
                                                    select(ds=Fecha, y=ventas_online_uds,
                                                           otros_precio_medio, trafico_web_sesiones_totales))
                             return(m_ser)
                           })
    res_row <- data.frame(prior_scale_1=prior_scale_1,
                          prior_scale_2=prior_scale_2)
    res_row <- cbind(res_row, res)
    print(res_row)
    res_df1 <- rbind(res_df1, res_row)
  }
}
res_df1
res_df1[res_df1$mape <= min(res_df1$mape)*1.1 &
          res_df1$mape >= min(res_df1$mape)*0.9,]
write.csv(res_df1, file="res_df1.csv", row.names=FALSE)


# 2. CV grid piors 2 variables explicativas + seasonality
res_df2 <- c()
for (prior_scale_1 in c(0.01, 0.1, 1, 5, 10)) {
  for (prior_scale_2 in c(0.01, 0.1, 1, 5, 10)) {
    for (seasonality_prior_scale in c(0.1, 1, 5, 10, 20)) {
      for (changepoint_prior_scale in c(0.01, 0.05, 0.1, 0.5, 1)) {
        for (monthly_prior_scale in c(0.1, 1, 5, 10, 20)) {
          res <- prophet.cv.loop(train %>%
                                   select(ds=Fecha, y=ventas_online_uds,
                                          otros_precio_medio, trafico_web_sesiones_totales),
                                 initial=106*7,
                                 period=14,
                                 horizon=28,
                                 units="days",
                                 model.function=function(data_train) {
                                   m_ser <- prophet(yearly.seasonality=10,
                                                    weekly.seasonality=FALSE,
                                                    daily.seasonality=FALSE,
                                                    seasonality.prior.scale=seasonality_prior_scale, #10
                                                    changepoint.prior.scale=changepoint_prior_scale) #0.05
                                   m_ser <- add_seasonality(m_ser, name='monthly', period=30.5,
                                                            fourier.order=8, prior.scale=monthly_prior_scale) #10
                                   m_ser <- add_regressor(m_ser, 'otros_precio_medio', 
                                                          prior.scale=prior_scale_1) #5
                                   m_ser <- add_regressor(m_ser, 'trafico_web_sesiones_totales', 
                                                          prior.scale=prior_scale_2) #5
                                   m_ser <- fit.prophet(m_ser, 
                                                        train %>%
                                                          select(ds=Fecha, y=ventas_online_uds,
                                                                 otros_precio_medio, trafico_web_sesiones_totales))
                                   return(m_ser)
                                 })
          res_row <- data.frame(prior_scale_1=prior_scale_1,
                                prior_scale_2=prior_scale_2,
                                seasonality_prior_scale=seasonality_prior_scale,
                                changepoint_prior_scale=changepoint_prior_scale,
                                monthly_prior_scale=monthly_prior_scale)
          res_row <- cbind(res_row, res)
          print(res_row)
          res_df2 <- rbind(res_df2, res_row)
        }
      }
    }
  }
}
res_df2
res_df2[res_df2$mape <= min(res_df2$mape)*1.001 &
          res_df2$mape >= min(res_df2$mape)*0.999,]
write.csv(res_df2, file="res_df2.csv", row.names=FALSE)
