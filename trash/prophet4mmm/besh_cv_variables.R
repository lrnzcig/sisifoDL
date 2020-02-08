library(prophet)
library(dplyr)

# punto de partida: modelo prophet con un solo regresor
m_ser <- prophet(yearly.seasonality=10,
                 weekly.seasonality=FALSE,
                 daily.seasonality=FALSE,
                 seasonality.prior.scale=10, #10
                 changepoint.prior.scale=0.05) #0.05
m_ser <- add_seasonality(m_ser, name='monthly', period=30.5,
                         fourier.order=8, prior.scale=10) #10
m_ser <- add_regressor(m_ser, 'trafico_web_sesiones_totales', prior.scale=5) #5
m_ser <- fit.prophet(m_ser, 
                     train %>%
                       select(ds=Fecha, y=ventas_online_uds,
                              trafico_web_sesiones_totales))
forecast_ser <- predict(m_ser, test %>%
                          select(ds=Fecha,
                                 trafico_web_sesiones_totales))

plot_preds_prophet(m_ser, forecast_ser, test, 
                   title="prophet mejor ajuste")
prophet_plot_components(m_ser, forecast_ser)


# bucle CV para buscar la siguiente mejor variable explicativa
res_df_var1 <- c()
for (variable_name in names(tb_weekly_base)[! names(tb_weekly_base) %in% c('Fecha', 
                                                                           'trafico_web_sesiones_totales') &
                                            ! grepl("^ventas_", names(tb_weekly_base))]) {
    res <- prophet.cv.loop(tb_weekly_base %>% 
                             mutate(Fecha=as.POSIXct(Fecha)) %>%
                             select(ds=Fecha, y=ventas_online_uds,
                                    !!variable_name, trafico_web_sesiones_totales),
                           horizon=13,
                           number_of_folds=1,
                           model.function=function(data_train) {
                             m_ser <- prophet(yearly.seasonality=10,
                                              weekly.seasonality=FALSE,
                                              daily.seasonality=FALSE,
                                              seasonality.prior.scale=10, #10
                                              changepoint.prior.scale=0.05) #0.05
                             m_ser <- add_seasonality(m_ser, name='monthly', period=30.5,
                                                      fourier.order=8, prior.scale=10) #10
                             m_ser <- add_regressor(m_ser, variable_name, 
                                                    prior.scale=5) #5
                             m_ser <- add_regressor(m_ser, 'trafico_web_sesiones_totales', 
                                                    prior.scale=5) #5
                             m_ser <- fit.prophet(m_ser, 
                                                  train %>%
                                                    select(ds=Fecha, y=ventas_online_uds,
                                                           !!variable_name, trafico_web_sesiones_totales))
                             return(m_ser)
                           },
                           do.plot=FALSE,
                           verbose=FALSE,
                           debug=FALSE)
    print(paste0("variable_name: ", variable_name,
                 ": mape: ", mean(res$mape)))
    res_row <- data.frame(variable_name=variable_name,
                          mean_mape=mean(res$mape))
    res_df_var1 <- rbind(res_df_var1, res_row)
}
res_df_var1
res_df_var1[res_df_var1$mean_mape <= min(res_df_var1$mean_mape)*1.2 &
              res_df_var1$mean_mape >= min(res_df_var1$mean_mape)*0.8,]
write.csv(res_df_var1, file="res_df_var1.csv", row.names=FALSE)



# resultado
m_ser <- prophet(yearly.seasonality=10,
                 weekly.seasonality=FALSE,
                 daily.seasonality=FALSE,
                 seasonality.prior.scale=10, #10
                 changepoint.prior.scale=0.05) #0.05
m_ser <- add_seasonality(m_ser, name='monthly', period=30.5,
                         fourier.order=8, prior.scale=10) #10
m_ser <- add_regressor(m_ser, 'trafico_web_sesiones_totales', prior.scale=5) #5
m_ser <- add_regressor(m_ser, 'goog_trends', prior.scale=5) #5
m_ser <- fit.prophet(m_ser, 
                     train %>%
                       select(ds=Fecha, y=ventas_online_uds,
                              trafico_web_sesiones_totales, goog_trends))
forecast_ser <- predict(m_ser, test %>%
                          select(ds=Fecha,
                                 trafico_web_sesiones_totales, goog_trends))

plot_preds_prophet(m_ser, forecast_ser, test, 
                   title="prophet mejor ajuste")
prophet_plot_components(m_ser, forecast_ser)
