##########
# 2.5 - The built-in distributions in R
##########

# For each distribution, there are 4 items that can be calculated: 
#density/point probability. cumulative probability, quantiles, and pseudo-random numbers

# Densities, AKA Probability Density Functions (PDFs)
x <- seq(-4,4,0.1)
plot(dnorm(x), type='l')
# Densities of discrete distributions are usually drawn as pin diagrams:
x <- 0:50
plot(x, dbinom(x, size=50, prob=0.33), type='h')

# Cumulated probability, AKA CDFs
x <- seq(-4,4,0.1)
plot(pnorm(x), type='l')

# Quantiles - inverse of CDF
# The p-quantile is the value with the property that there is a probability p of getting a value less than or equal than it
# Median = 50% quantile
probs <- seq(0,1,0.1)
plot(qnorm(probs), type='l')

# Generating random numbers from a given distribution
# Normal distribution
rnorm(10,mean=5,sd=2)
# Uniform distribution
runif(10)
# Binomial distribution
rbinom(10,size=20,prob=0.5)
