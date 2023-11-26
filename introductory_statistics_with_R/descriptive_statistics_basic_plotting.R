##########
# 4 - Descriptive Statistics and graphics
##########

# Basic summary statistics are easy:
x <- rnorm(50)
mean(x)
sd(x)
var(x) # sd squared
median(x)
quantile(x) # QUANTiles = 0,0.25.0.5,0.75,1 - DECiles = 0.1,0.2,...,0.2,1, (per)CENTiles = 0.01,0.02,...,0.99,1
# Getting other quantiles, first generate vector of desired quantiles:
v <- seq(0,1,0.1)
quantile(x,v)

summary(x)


## Histograms
hist(x)
hist(x, breaks=c(-3,2,1,3)) # Have full control over where breaks are when passed as vector. Otherwise, it is just approx. n bars

## Empirical Cumulative Distribution
n <- length(x) # N = length of data
plot(x=sort(x), y=(1:n)/n, type='s')
ecdf(x)

## Q-Q Plots
# Plots the kth smallest observation against the expected value of the kth smallest observations out of n in a standard Normal Distribution
qqnorm(x)

## Boxplots
# A layout with 2 plots side by side is specified using the mfrow graphical parameter (MultiFrame ROWwise, 1x2 layout)
# mfcol performs the same operations, but COLUMN-wise
par(mfrow=c(1,2))
boxplot(x)

## Grouped statistics
library(ISwR)
attach(red.cell.folate) # Attaches object(s)/database to the R search path
# (value to aggregate, value to group by, aggregate function)
tapply(folate, ventilation, mean)
tapply(folate, ventilation, sd)
tapply(folate, ventilation, length)
# tapply can work with NaN data:
attach(juul)
tapply(igf1, tanner, mean, na.rm=T)
# aggregate() is for same purpose, but works on entire dataframe presents data as a dataframe
aggregate(juul[c('age','igf1')], list(sex=juul$sex), mean, na.rm=T)
