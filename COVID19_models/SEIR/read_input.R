
getData <- function(data_path="RL_input") {
  setwd(data_path)
  
  thetas_init <- as.matrix(read.csv("parameters.csv"))
  thetas_sd_init <- as.matrix(read.csv("parameters_sd.csv"))
  states <- as.matrix(read.csv("states.csv"))
  time_series_predictions <- as.matrix(read.csv("time_series_predictions.csv"))
  
  return(list(thetas_init=thetas_init, 
              thetas_sd_init=thetas_sd_init, 
              latent=5.1, 
              gamma=0.06, 
              states=states,
              time_series_predictions=time_series_predictions))
}