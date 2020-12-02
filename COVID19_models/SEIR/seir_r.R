###This code is to be used in RL
###The RL should just concern the reduction_control; other inputs will be either from Bayesian updating or given as data.
###Output is the number of infected people by the end of prediction window (from current to pred)

### Arguements description:
## reduction_control:reductions in grocery shopping, retail shopping, and workplace activities. It is a matrix with 3 columns and the same number of rows as the number of days (from current to Pred). 
## reduction_time_series: reductions in park recreation and residential activities. It is a matrix with 2 columns and the same number of rows as the number of days (from current to Pred).
## thetas_data: estimated mean values of thetas from Bayesian updating. It is a matrix with 6 columns and 114 rows (from Jan 22 to May 15).
## thetas_sd_data: estimated standard deviation values of thetas from Bayesian updating. It is a matrix with 6 columns and 114 rows (from Jan 22 to May 15).
## latent: length of latent period. It is a scalar.
## gamma: recovery rate. It is a scalar.
## St_data, Et_data, It_data, Rt_data: daily numbers of susceptible, exposed, infectious and removed people from Jan 22 onward. Each is a vector.
## popu: population size. It is a scalar.
## current: starting date of prediction time window.
## pred: end date of prediction time window.
seirPredictions <- function(reduction_control, reduction_time_series, thetas_data, thetas_sd_data, latent, gamma, St_data, Et_data, It_data, Rt_data, popu, current, pred) {
  
  thetas <- beta_data[current,]
  thetas_sd <- beta_sd_data[current,]
  St <- St_data[current]
  Et <- Et_data[current]
  It <- It_data[current]
  Rt <- Rt_data[current]
  
  oneTimeStep <- function(theta_l, latent, gamma, S, E, I, R, popu, t_step) {
    theta_new <- rnorm(length(theta_l),mean=theta_l,sd=theta_sd)
    reduction_new <- c(reduction_control[t_step,], reduction_time_series[t_step,])
    beta_new <- exp(theta_new[1] + sum(theta_new[2:6]*reduction_new))
    
    delta_S <- -beta_new * I * S / popu
    delta_E <-  beta_new * I * S / popu - E * latent
    delta_I <-  E * latent - I * gamma
    delta_R <-  I * gamma
    
    S_new <- S + delta_S
    E_new <- E + delta_E
    I_new <- I + delta_I
    R_new <- R + delta_R
    
    return(list(theta_new, S_new, E_new, I_new, R_new))
  }
  
  for (time_step in 1:pred) {
    predictions <- oneTimeStep(thetas, latent, gamma, St, Et, It, Rt, popu, time_step)
    thetas <- predictions[[1]]
    St <- predictions[[2]]
    Et <- predictions[[3]]
    It <- predictions[[4]]
    Rt <- predictions[[5]]
  }
  
  cum_infections <- Et+It+Rt
  
  return(cum_infections)
}
