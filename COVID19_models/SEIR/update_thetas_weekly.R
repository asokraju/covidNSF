### This function calculate the values of thetas and thetas' standard deviations from (current+1) to pred.
### The function returns a matrix with 6 rows (each corresponding to one theta) and the same number of rows as the number of weeks (counting from the week May 10-16 as week 1).

### Arguement description:
## thetas_so_far: values of thetas for the week May 3-9. It is a vector with 6 elements (each corresponding to one theta).
## thetas_sd_so_far: values of thetas's standard deviations for the week May 3-9. It is a vector with 6 elements (each corresponding to one theta).
## pred: the week index (counting from the week May 10-16 as week 1) of the end of the prediction time window.

getThetas <- function(thetas_so_far, thetas_sd_so_far, pred) {

  thetas_updated <- matrix(NA,pred+1,6)
  thetas_sd_updated <- matrix(NA, 1,6)
  thetas_sd_updated[1,] <- thetas_sd_so_far
  thetas_updated[1,] <- thetas_so_far

  for (ti in 1:pred) {

    thetas_ti <- rnorm(6, mean=thetas_updated[ti,], sd=thetas_sd_updated[1,])
    thetas_updated[ti+1,] <- thetas_ti
  }

  thetas_updated <- thetas_updated[-1,]

  return(thetas_updated)
}
