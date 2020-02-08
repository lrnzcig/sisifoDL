# aproximación a prophet en MMM imitando un modelo ajustado por LM en Bershka
# el ajuste de los priors se hace manualmente
# necesita precargar RDS con datos de Bershka


source("utils.R")

names(tb_weekly_base)


# 0. referencia con lm_fit_ventas_online_uds
# ... variable target
var_tgt_online_uds

# ... variables explicativas
vars_model_online_uds

tb_weekly_base$Fecha %>% head()
tb_weekly_base$Fecha %>% tail()

tb_weekly_base %>% select(vars_model_online_uds) %>% head()

# ... train/test
# - prediccion a 4 pasos
# - 4 folds de CV
# - 2 folds para validacion final
# train fecha inicio 2015-10-11
# train es 130-2*4=122, fecha final es 2018-02-04
# train del 1er fold de CV es 130-6*4=106, fecha final es 2017-10-15 (por lo tanto aproximadamente 2 años)
# test va de 2018-02-11 a 2018-03-31
train <- tb_weekly_base %>% 
  filter(Fecha < "2018-02-11") %>%
  mutate(Fecha=as.POSIXct(Fecha))
test <- tb_weekly_base %>% 
  filter(Fecha >= "2018-02-11") %>%
  mutate(Fecha=as.POSIXct(Fecha))

# ... predicciones y MAPE
ref_preds <- predict(lm_fit_ventas_online_uds, test)
ref_preds_df <- data.frame(ds= test %>%
                             select(ds=Fecha),
                           p=ref_preds)
plot_preds(ref_preds_df, test,
           title="modelo LM referencia")




# 1. prophet para el mismo df sin variables explicativas 
# (estacionalidad por defecto, changepoints tendencia no tienen sentido)
m_s <- prophet(train %>%
                 select(ds=Fecha, y=ventas_online_uds))
               #changepoint.prior.scale = 0.5)

forecast_s <- predict(m_s, tb_weekly_base %>%
                        select(ds=Fecha))

plot_preds_prophet(m_s, forecast_s, test, 
                   title="prophet sin variables explicativas")
prophet_plot_components(m_s, forecast_s)





# 2. prophet sin variables explicativas con estacionalidad
# TODO custom para festivos, country_holidays no funciona bien
m_se <- prophet(yearly.seasonality=10, # 10 es valor por defecto y parece el mejor
                weekly.seasonality=FALSE,
                daily.seasonality=FALSE,
                seasonality.prior.scale=10, #10
                changepoint.prior.scale=0.01) #0.05
m_se <- add_seasonality(m_se, name='monthly', period=30.5, 
                        fourier.order=8, # mejor que 5 pero habría que investigar
                        prior.scale=20) #20
#m_se <- add_country_holidays(m_se, country_name = 'ES')
m_se <- fit.prophet(m_se, 
                    train %>%
                      select(ds=Fecha, y=ventas_online_uds))
forecast_se <- predict(m_se, tb_weekly_base %>%
                         select(ds=Fecha))

plot_preds_prophet(m_se, forecast_se, test, 
                   title="prophet sin variables explicativas con estacionalidad")
prophet_plot_components(m_se, forecast_se)





# 3. prophet con regresores
m_ser <- prophet(yearly.seasonality=10,
                 weekly.seasonality=FALSE,
                 daily.seasonality=FALSE,
                 seasonality.prior.scale=5, #10
                 changepoint.prior.scale=0.01) #0.05
m_ser <- add_seasonality(m_ser, name='monthly', period=30.5,
                         fourier.order=8, prior.scale=20) #10
m_ser <- add_regressor(m_ser, 'otros_precio_medio', prior.scale=0.1) #5
m_ser <- add_regressor(m_ser, 'trafico_web_sesiones_totales', prior.scale=0.1) #5
m_ser <- fit.prophet(m_ser, 
                     train %>%
                       select(ds=Fecha, y=ventas_online_uds,
                              otros_precio_medio, trafico_web_sesiones_totales))
forecast_ser <- predict(m_ser, tb_weekly_base %>%
                          select(ds=Fecha,
                                 otros_precio_medio, trafico_web_sesiones_totales))

plot_preds_prophet(m_ser, forecast_ser, test, 
                   title="prophet mejor ajuste")
prophet_plot_components(m_ser, forecast_ser)


# 3.1. cross-validation
df.cv_ser <- cross_validation(m_ser, initial = 106*7, period = 14, horizon = 28, units="days")
df.cv_ser %>% as.data.frame()
df.p_ser <- performance_metrics(df.cv_ser)
# ... el resultado de coverage es malo; calculado con un intervalo de confianza del 80% (valor por defecto)
# ...   y solo para la tendencia
df.p_ser





# 3.2 intervalos confianza componentes
m_ser_mcmc <- prophet(yearly.seasonality=10,
                      weekly.seasonality=FALSE,
                      daily.seasonality=FALSE,
                      seasonality.prior.scale=5, #10
                      changepoint.prior.scale=0.01, #0.05
                      mcmc.samples=300,
                      interval.width=0.95)
m_ser_mcmc <- add_seasonality(m_ser_mcmc, name='monthly', period=30.5,
                              fourier.order=8, prior.scale=20) #10
m_ser_mcmc <- add_regressor(m_ser_mcmc, 'otros_precio_medio', prior.scale=0.1) #5
m_ser_mcmc <- add_regressor(m_ser_mcmc, 'trafico_web_sesiones_totales', prior.scale=0.1) #5
m_ser_mcmc <- fit.prophet(m_ser_mcmc, 
                          train %>%
                            select(ds=Fecha, y=ventas_online_uds,
                                   otros_precio_medio, trafico_web_sesiones_totales))
forecast_ser_mcmc <- predict(m_ser_mcmc, tb_weekly_base %>%
                               select(ds=Fecha,
                                      otros_precio_medio, trafico_web_sesiones_totales))
prophet_plot_components(m_ser_mcmc, forecast_ser_mcmc)

# ... el resultado de coverage mejora al aumentar el intervalo de confianza a 95% e incluir estacionalidad
df.cv_ser_mcmc <- cross_validation(m_ser_mcmc, initial = 106*7, period = 14, horizon = 28, units="days")
df.cv_ser_mcmc %>% as.data.frame()
df.p_ser_mcmc = performance_metrics(df.cv_ser_mcmc)
df.p_ser_mcmc





# 4. prophet con regresores resultado de CV loop
# TODO coger valores con changepoint.prior.scale más bajo
m_ser_opt <- prophet(yearly.seasonality=10,
                     weekly.seasonality=FALSE,
                     daily.seasonality=FALSE,
                     seasonality.prior.scale=0.1,#20
                     changepoint.prior.scale=0.5)
m_ser_opt <- add_seasonality(m_ser_opt, name='monthly', period=30.5,
                             fourier.order=8, prior.scale=0.1)#5
m_ser_opt <- add_regressor(m_ser_opt, 'otros_precio_medio', prior.scale=10) #0.01
m_ser_opt <- add_regressor(m_ser_opt, 'trafico_web_sesiones_totales', prior.scale=0.1)
m_ser_opt <- fit.prophet(m_ser_opt, 
                         train %>%
                           select(ds=Fecha, y=ventas_online_uds,
                                  otros_precio_medio, trafico_web_sesiones_totales))
forecast_ser_opt <- predict(m_ser_opt, tb_weekly_base %>%
                              select(ds=Fecha,
                                     otros_precio_medio, trafico_web_sesiones_totales))

plot_preds_prophet(m_ser_opt, forecast_ser_opt, test, 
                   title="prophet mejor ajuste")
prophet_plot_components(m_ser_opt, forecast_ser)
