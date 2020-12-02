###This code is to be used in RL
###The RL should just concern the reduction_control; other inputs will be either from Bayesian updating or given as data.
###Output is the states by the end of prediction window (specified by current and pred)

### Arguements description:
## reduction_control:reductions in grocery shopping, retail shopping, and workplace activities. It is a matrix with 3 columns and the same number of rows as the number of weeks (from the week of May 3-9 to end of prediction window; extra rows will be fine). 
## reduction_time_series: reductions in park recreation and residential activities. It is a matrix with 2 columns and the same number of rows as reduction_control.
## thetas_data: estimated mean values of thetas from Bayesian updating. It is a matrix with 6 columns and number of rows same as reduction_control.
## latent: length of latent period. It is a scalar.
## gamma: recovery rate. It is a scalar.
## St_data, Et_data, It_data, Rt_data: weekly numbers of susceptible, exposed, infectious and removed people from May 3-9 onward. Each is a vector.
## popu: population size. It is a scalar.
## current: starting week of prediction time window (counting the week of May 3-9 as week 1).
## pred: end week of prediction time window (counting the week of May 3-9 as week 1).

### Note:
## please correspond thetas with relevant reduction variables
## if control is done week by week with moving current day, pred should be set to 1.
seirPredictions <- function(reduction_control, reduction_time_series, thetas_data, latent, gamma, St_data, Et_data, It_data, Rt_data, popu, current, pred) {
  
  St <- St_data[current]
  Et <- Et_data[current]
  It <- It_data[current]
  Rt <- Rt_data[current]

  oneTimeStep <- function(latent, gamma, S, E, I, R, popu, t_step) {

    thetas_data <- matrix(thetas_data,1,6)
    reduction_control <- matrix(reduction_control,1,3)
    reduction_time_series <- matrix(reduction_time_series,1,2)

    theta_new <- thetas_data[t_step,]   #rnorm(length(theta_l),mean=theta_l,sd=theta_sd)
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
    
    return(list(S_new, E_new, I_new, R_new, beta_new))
  }
  
  for (time_step in (current):(pred)) {
    predictions <- oneTimeStep(latent, gamma, St, Et, It, Rt, popu, time_step)
    St <- predictions[[1]]
    Et <- predictions[[2]]
    It <- predictions[[3]]
    Rt <- predictions[[4]]
    beta_new <-predictions[[5]]
  }
  
  cum_infections <- Et+It+Rt
  
  return(list(St, Et, It, Rt, beta_new))
}
