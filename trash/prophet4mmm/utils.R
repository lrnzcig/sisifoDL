library(ggplot2)
library(dplyr)
library(prophet)


#' Obtener estimación del Beta de una varialbe a partir de resultados de propeht
#' 
#' @param data dataframe
#' @param forecast resultado de prophet
#' @param variable nombre de la variable
#' @return beta
get_coef <- function(data, forecast, variable){
  (data %>% select(ds, data:= !!enquo(variable)) %>%  
     left_join(forecast %>% 
                 mutate(ds = as.Date(ds)) %>% 
                 select(ds, ap := !!enquo(variable)), by = "ds") %>% 
     lm("ap ~ data", data = .))$coefficients["data"]
} 



#' plot de las predicciones a partir de dfs para real y predicciones
#' 
#' @param preds_df dataframe con predicciones
#' @param test_df dataframe de test
#' @param title titular del plot
#' @return ggplot
plot_preds <- function(preds_df, test_df, title) {
  plot_preds_internal(ggplot(data=preds_df, aes(ds, p)) + 
                        geom_line(color="blue"),
                      preds_df, test_df, title)
}


#' función interna, no utilizar
plot_preds_internal <- function(ggplot_object, 
                                preds_df, test_df, title) {
  real <- test_df %>%
    select(ds=Fecha, y=ventas_online_uds) # hardcode
  
  error <- real$y - preds_df$p 
  mape <- mean(abs(error) / real$y)
  
  ggplot_object +
    geom_line(data=real, 
              aes(ds, y),
              color="orange") +
    ggtitle(title, paste0("MAPE=", scales::percent(mape))) +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
}

#' plot de las predicciones a partir de df para test y prophet para predicciones
#' 
#' @param m modelo prophet
#' @param forecast objeto de prophet
#' @param test_df dataframe de test
#' @param title titular del plot
#' @return ggplot
plot_preds_prophet <- function(m, forecast, test_df, title) {
  preds_df <- forecast %>% 
    filter(ds >= min(test_df$Fecha)) %>%
    select(ds, p=yhat)
  plot_preds_internal(plot(m, forecast) + 
                        add_changepoints_to_plot(m),
                      preds_df, test_df, title)
}



#' cross-validation loop
#' ajusta modelo con forward chaining y obtiene métricas de precisión
#' 
#' @param data daframe completo (train+test)
#' @param horizon horizonte de la predicción (número de pasos)
#' @param number_of_folds número de particiones para forward chaining
#' @param model.funcion función que ajusta el modelo a partir de df de entrenamiento
#' @param do.plot si a TRUE, hace plot de predicciones
#' @param verbose si a TRUE muestra mensajes
#' @param debug si a TRUE muestra mensajes para depuración, difíciles de entender
#' @return dataframe con MAPEs de cada uno de los folds
prophet.cv.loop <- function(data_train,
                            initial,
                            period,
                            horizon,
                            units,
                            model.function) {
  # fit model & predict & evaluate
  model <- model.function(data_train)
  df.cv <- cross_validation(model, initial=initial, period=period, horizon=horizon, units=units)
  df.p <- performance_metrics(df.cv)
  return(df.p[nrow(df.p),]) # devolver última fila
}
