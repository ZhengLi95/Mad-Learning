### TOPICS:
### Linear models

library(MASS)
library(car)
library(rgl)

#_______________________________________________________________________________
##### Linear models
#####---------------

### Example 1: Multiple linear regression
###----------------------------------------
### Dataset cars: distance taken to stop [ft] as a function of velocity [mph]
### for some cars in the 1920s

cars
help(cars)

x11()
plot(cars, xlab = 'Speed (mph)', ylab = 'Stopping distance (ft)', las = 1)
# Q1: is this a linear or multiple regression?

n          <- dim(cars)[[1]]
distance   <- cars$dist
speed1     <- cars$speed
speed2     <- cars$speed^2

### Model:
### distance = beta_0 + beta_1 * speed + beta_2 * speed^2 + Eps
### (linear in the parameters!)

### Assumptions
## 1) Parameter estimation: E(Eps) = 0  and  Var(Eps) = sigma^2 
## 2) Inference:            Eps ~ N(0, sigma^2)


##### 1) Estimate of the parameters
##### Assumptions: E(Eps) = 0  and  Var(Eps) = sigma^2 
#####--------------------------------------------------

fm <- lm(distance ~ speed1 + speed2)

summary(fm) 
# Note: collinearity

fitted(fm)        # y hat
residuals(fm)     # eps hat

coefficients(fm)  # beta_i
vcov(fm)          # cov(beta_i)

fm$rank # order of the model [r+1]
fm$df   # degrees of freedom of the residuals [n-(r+1)]

hatvalues(fm) # h_ii
rstandard(fm) # standardized residuals

s2_err <- sum(residuals(fm)^2)/fm$df  # estimate of sigma^2

x11()
plot(cars, xlab = 'Speed (mph)', ylab = 'Stopping distance (ft)', las = 1, xlim=c(0,30), ylim=c(-5,130))
x <- seq(0,30,by=0.1)
b <- coef(fm)
lines(x, b[1]+b[2]*x+b[3]*x^2)

##### Inference on the parameters
##### Assuption: Eps ~ N(0, sigma^2)
#####-------------------------------------------
### Test (Fisher):
# H0: (beta1, beta2) == (0, 0) vs H1: (beta1, beta2) != (0, 0)
linearHypothesis(fm, rbind(c(0,1,0), c(0,0,1)), c(0,0)) 
summary(fm)

p <- 2  # number of tested coefficients
r <- 2  # number of regressors

# Confidence region:
# center: point estimate
c(coefficients(fm)[2], coefficients(fm)[3])
# Direction of the axes?
x11()
plot(coefficients(fm)[2], coefficients(fm)[3], xlim = c(-7,7), ylim = c(-7,7), asp=1, xlab='beta1', ylab='beta2')
ellipse(coefficients(fm)[2:3], vcov(fm)[2:3,2:3], sqrt(p * qf(1 - 0.05, p, n-(r+1))))   # ellisse di confidenza al 95%
abline(v=0)
abline(h=0)
# Note: collinearity!

# Bonferroni intervals (level 95%)
Bf <- rbind(
  beta1=c(coefficients(fm)[2] - sqrt(vcov(fm)[2,2]) * qt(1 - 0.05/(2*p), n-(r+1)),
          coefficients(fm)[2] + sqrt(vcov(fm)[2,2]) * qt(1 - 0.05/(2*p), n-(r+1))),
  beta2=c(coefficients(fm)[3] - sqrt(vcov(fm)[3,3]) * qt(1 - 0.05/(2*p), n-(r+1)),
          coefficients(fm)[3] + sqrt(vcov(fm)[3,3]) * qt(1 - 0.05/(2*p), n-(r+1)))
)
Bf

# or (only for intervals on beta)
confint(fm, level= 1-0.05/p)[2:3,]  # Bonferroni correction!

# Note: confint() returns the confidence intervals one-at-a-time;
# to have a global level 95% we need to include a correction

### Test:
# H0: (beta0+beta2, beta1) == (0, 0) vs H1: (beta0+beta2, beta1) != (0, 0)
C <- rbind(c(1,0,1), c(0,1,0))
linearHypothesis(fm, C, c(0,0))

# Homework
# Build the associated confidence region

##### Confidence intervals for the mean
##### & prediction (new obs)
##### Assumption: Eps ~ N(0, sigma^2)
#####---------------------------------
# Command predict()
Z0.new <- data.frame(speed1=10, speed2=10^2)

# Conf. int. for the mean
Conf <- predict(fm, Z0.new, interval='confidence', level=1-0.05)  
Conf
# Pred. int. for a new obs
Pred <- predict(fm, Z0.new, interval='prediction', level=1-0.05)  
Pred

x11()
plot(cars, xlab = 'Speed (mph)', ylab = 'Stopping distance (ft)', las = 1, xlim=c(0,30), ylim=c(-5,130))
x <- seq(0,30,by=0.1)
b <- coef(fm)
lines(x, b[1]+b[2]*x+b[3]*x^2)
points(10,Conf[1], pch=19)
segments(10,Pred[2],10,Pred[3],col='gold', lwd=2)
segments(10,Conf[2],10,Conf[3],col='red', lwd=2)
points(10,Conf[2], pch='-', col='red', lwd=2)
points(10,Conf[3], pch='-', col='red', lwd=2)
points(10,Pred[2], pch='-', col='gold', lwd=2)
points(10,Pred[3], pch='-', col='gold', lwd=2)

dev.off()

# We can repeat these for values of speed between 0 and 30
# (point-wise intervals!)
Z0   <- data.frame(cbind(speed1=seq(0, 30, length=100), speed2=seq(0, 30, length=100)^2))
Conf <- predict(fm, Z0, interval='confidence')
Pred <- predict(fm, Z0, interval='prediction')

x11()
plot(cars, xlab = 'Speed (mph)', ylab = 'Stopping distance (ft)', las = 1, xlim=c(0,30), ylim=c(-45,150))
lines(Z0[,1], Conf[,'fit'])
lines(Z0[,1], Conf[,'lwr'], lty=2, col='red', lwd=2)
lines(Z0[,1], Conf[,'upr'], lty=2, col='red', lwd=2)

lines(Z0[,1], Pred[,'lwr'], lty=3, col='gold', lwd=2)
lines(Z0[,1], Pred[,'upr'], lty=3, col='gold', lwd=2)

dev.off()

### Exercise: how can we obtain a confidence/prediction band?
### Hint: use T2 simultaneous intervals

##### Verify assumptions
#####--------------------
x11()
par(mfrow=c(2,2))
plot(fm)

shapiro.test(residuals(fm))

graphics.off()

### Q2: what happens if we change unit of measure?
### Exercise: Compare the results obtained from:
distance.m      <- cars$dist*0.3         # feet -> m
speed1.kmh      <- cars$speed*1.6        # miles/hour -> km per hour
speed2.kmh2     <- cars$speed^2 * 1.6^2
### Compare:
### - Coefficient estimates and corresponding covariance matrices
### - confidence intervals on the coefficients
### - fitted values
### - residuals
### - results of the tests
###    H0: (beta1, beta2) == (0, 0) vs H1: (beta1, beta2) != (0, 0)
###    H0: (beta0+beta2, beta1) == (0, 0) vs H1: (beta0+beta2, beta1) != (0, 0)

#_______________________________________________________________________________
### Example 2: Anscombe
###---------------------------------

### This example show some cases for which examining data and 
### residuals is crucial (don't look at R2 only!)

anscombe
# Four x-y datasets which have the same traditional statistical 
# properties (mean, variance, correlation, regression line, etc.), 
# yet are quite different.

attach(anscombe)

# dataset 1
lm1 <- lm(y1~ x1)
summary(lm1)

x11(width=14, height=7)
par(mfcol=c(2,4))
plot(x1,y1, main='Dataset 1')
abline(lm1)

plot(x1,residuals(lm1))
abline(h=0)

# dataset 2
lm2 <- lm(y2~ x2)
summary(lm2)

plot(x2,y2, main='Dataset 2')
abline(lm2)

plot(x2,residuals(lm2))
abline(h=0)

# same R^2, same coefficient estimate, same residual std error

# dataset 3
lm3 <- lm(y3~ x3)
summary(lm3)

plot(x3,y3, main='Dataset 3')
abline(lm3)

plot(x3,residuals(lm3))
abline(h=0)

# dataset 4
lm4 <- lm(y4~ x4)
summary(lm4)

plot(x4,y4, main='Dataset 4')
abline(lm4)

plot(x4,residuals(lm4))
abline(h=0)

graphics.off()
detach(anscombe)

#_______________________________________________________________________________
### Example 3: Earthquakes
###---------------------------------
Q <- cbind(quakes[,1:2], depth=-quakes[,3]/100)

d <- dist(Q)
clusterw <- cutree(hclust(d, method='ward.D2'), 2)

open3d()
points3d(x=Q$lat, y=Q$long, z=Q$depth, size=4, col=clusterw+1, aspect = T)
box3d()
axes3d()

Q1 <- Q[clusterw=='1',]
Q2 <- Q[clusterw=='2',]
Q  <- Q

### Model 1 (all together)
###-----------------------
# Model:
# depth = beta0 + beta1*lat + beta2*long + eps

fit  <- lm(depth ~ lat + long, data=Q)
summary(fit)

points3d(x=Q$lat, y=Q$long, z=fitted(fit), size=4)
surface3d(range(Q$lat), range(Q$long), 
          matrix(predict(fit, expand.grid(lat = range(Q$lat), long = range(Q$long))),2,2),
          alpha = 0.5)

x11()
par(mfrow=c(2,2))
plot(fit)

shapiro.test(rstandard(fit))

dev.off()

### Model 2 (dummy variable)
###--------------------------
dummy <- clusterw - 1
# 0 = red
# 1 = green

Qd <- cbind(Q, dummy)

# Model:
# depth = beta0       + beta1*lat       + beta2*long        +
#       + beta3*dummy + beta4*dummy*lat + beta5*dummy*long  + eps
# i.e.,
# depth = B0[g] + B1[g]*lat + B2[g]*long + eps
# with B0[g]=beta0       if the unit is in group s.t. dummy=0 (red)
#      B0[g]=beta0+beta3 if the unit is in group s.t. dummy=1 (green)
#      B1[g]=beta1       if the unit is in group s.t. dummy=0 (red)
#      B1[g]=beta1+beta4 if the unit is in group s.t. dummy=1 (green)
#      B2[g]=beta2       if the unit is in group s.t. dummy=0 (red)
#      B2[g]=beta2+beta5 if the unit is in group s.t. dummy=1 (green)

fitd <- lm(depth ~ lat + long + dummy + lat:dummy + long:dummy, data=Qd)
summary(fitd)

# Fitted model:
open3d()
points3d(x=Q$lat, y=Q$long, z=Q$depth, size=4, col=clusterw+1, aspect = T)
points3d(x=Qd$lat, y=Qd$long, z=fitted(fitd), size=4)

surface3d(range(Q$lat), range(Q$long), 
          matrix(predict(fitd, expand.grid(lat = range(Q$lat), long = range(Q$long), dummy=c(1,1))),2,2),
          alpha = 0.5, col='green')
surface3d(range(Q$lat), range(Q$long), 
          matrix(predict(fitd, expand.grid(lat = range(Q$lat), long = range(Q$long), dummy=c(0,0))),2,2),
          alpha = 0.5, col='red')
box3d()
axes3d()

# Residuals:
x11()
par(mfrow=c(2,2))
plot(fitd)
shapiro.test(rstandard(fitd))

dev.off()

# test: are the two planes needed?
A <- rbind(c(0,0,0,1,0,0), c(0,0,0,0,1,0), c(0,0,0,0,0,1))
b <- c(0,0,0)
linearHypothesis(fitd, A, b)

# Reduce the model:
summary(fitd)

### Model 3 (reduced model)
###--------------------------
# Model:
# depth = beta0       + beta1*lat       + beta2*long        +
#       + beta3*dummy + beta4*dummy*long  + eps
# i.e.,
# depth = B0[g] + B1*lat + B2[g]*long
# with B0[g]=beta0       if the unit is in group s.t. dummy=0 (red)
#      B0[g]=beta0+beta3 if the unit is in group s.t. dummy=1 (green)
#      B1=beta1
#      B2[g]=beta2       if the unit is in group s.t. dummy=0 (red)
#      B2[g]=beta2+beta5 if the unit is in group s.t. dummy=1 (green)

fitD <- lm(depth ~ lat + long + dummy + long:dummy, data=Qd)
summary(fitD)

# Fitted model
open3d()
points3d(x=Q$lat, y=Q$long, z=Q$depth, size=4, col=clusterw+1, aspect = T)
axes3d()
points3d(x=Qd$lat, y=Qd$long, z=fitted(fitD), size=4, col = 'blue')
surface3d(range(Q$lat), range(Q$long), 
          matrix(predict(fitD, expand.grid(lat = range(Q$lat), long = range(Q$long), dummy=c(1,1))),2,2),
          alpha = 0.5, col='green')
surface3d(range(Q$lat), range(Q$long), 
          matrix(predict(fitD, expand.grid(lat = range(Q$lat), long = range(Q$long), dummy=c(0,0))),2,2),
          alpha = 0.5, col='red')

# Residuals:
x11()
par(mfrow=c(2,2))
plot(fitD)
shapiro.test(rstandard(fitD))

### Model 4 (quadratic dependence)
###--------------------------------------

Fit  <- lm(depth ~ lat + long + I(lat^2) + I(long^2) + lat:long, data=Q)
summary(Fit)

# Fitted model
open3d()
points3d(x=Q$lat, y=Q$long, z=Q$depth, size=4, col=clusterw+1, aspect = T)
axes3d()
points3d(x=Q$lat, y=Q$long, z=fitted(Fit), size=4)

x11()
par(mfrow=c(2,2))
plot(Fit)

shapiro.test(rstandard(Fit))

dev.off()

# test: are the quadratic terms needed?
A <- rbind(c(0,0,0,1,0,0), c(0,0,0,0,1,0), c(0,0,0,0,0,1))
b <- c(0,0,0)
linearHypothesis(Fit, A, b)

# test: is the latitude needed?
A <- rbind(c(0,1,0,0,0,0), c(0,0,0,1,0,0), c(0,0,0,0,0,1))
b <- c(0,0,0)
linearHypothesis(Fit, A, b)

# test: is the longitude needed?
A <- rbind(c(0,0,1,0,0,0), c(0,0,0,0,1,0), c(0,0,0,0,0,1))
b <- c(0,0,0)
linearHypothesis(Fit, A, b)

#_______________________________________________________________________________
##### Example 4: Bias-variance trade-off 
#####------------------------------------

# generation of training set and test set
set.seed(28071987)

# true model: regressors x, x^2, x^3; coefficients (1,1,1,-1)
f <- function(x){1+x+x^2-x^3}
sigma <- 0.25
x <- seq(-1, 1.5, length = 21)
y <- f(x) + rnorm(21, sd = sigma)
y.new <- f(x) + rnorm(21, sd = sigma)

# build design matrix
data <- NULL
for(p in 0:20)
  data <- cbind(data, x^p)
colnames(data) <- c(paste('x', 0:20, sep=''))
data <- data.frame(data)
head(data)

# data to predict
data.new <- NULL
for(p in 0:20)
  data.new <- cbind(data.new, x^p)
colnames(data.new) <- c(paste('x', 0:20, sep=''))
data.new <- data.frame(data.new)
head(data.new)

# grid to plot
data.plot <- NULL
x.plot <- seq(-1, 1.5, length = 210)
for(p in 0:20)
  data.plot <- cbind(data.plot, x.plot^p)
colnames(data.plot) <- c(paste('x', 0:20, sep=''))
data.plot <- data.frame(data.plot)

# plot of the training set, test set and "true" mean curve
x11()
plot(x, y)                                    # training set
points(x, y.new, col='red')                   # test set
lines(x.plot, f(x.plot), col='blue', lty=2)   # true mean

# regression with polynomials of increasing order
SSres <- SSres.new <- s2 <- b <- R2 <- R2.adj <- NULL
n <- 21

x11(width=14, height=7)
par(mfrow=c(4,4), mar=rep(2,4))
for(p in 1:16)
{
  fit <- lm(y ~ 0 + . , data[,1:(p+1)])
  plot(x, y)
  points(x, y.new, col='red')
  lines(x.plot, predict(fit, data.plot))
  lines(x.plot, f(x.plot), col='blue', lty=2)
  
  SSres <- c(SSres, sum((y - fitted(fit))^2))
  SSres.new <- c(SSres.new, sum((y.new - fitted(fit))^2))
  s2 <- c(s2, sum((y - fitted(fit))^2)/(n - (p+1)))
  R2 <- c(R2, summary(fit)$r.squared)
  R2.adj <- c(R2.adj, summary(fit)$adj.r.squared)
  bp <- rep(0,17)
  bp[1:(p+1)] <- coefficients(fit)
  b <- cbind(b, bp)
}

# compare some indices

x11()
par(mfrow=c(2,2))
plot(1:16, SSres, pch=16)
abline(v=3, col='blue', lty=2)
plot(1:16, SSres.new, pch=16)
abline(v=3, col='blue', lty=2)
plot(1:16, R2, pch=16)
abline(v=3, col='blue', lty=2)
plot(1:16, R2.adj, pch=16)
abline(v=3, col='blue', lty=2)

# compare parameter estimates
# true model: regressors x, x^2, x^2; coefficients (1,1,1,-1)

b.vero <- c(1,1,1,-1, rep(0,8))

x11()
par(mfrow=c(2,3))
for(i in 1:5)
{
  plot(1:16, b[i,], ylim=c(-5, 5), main=paste('b',i-1,sep=''), pch=16)
  grid()
  abline(v=3, col='blue', lty=2)
  abline(h=b.vero[i], col='blue')
}
plot(1:16, s2, main='sigma^2', pch=16)
grid()
abline(v=3, col='blue', lty=2)
abline(h=sigma^2, col='blue')

graphics.off()

#_______________________________________________________________________________
##### Exercises on linear models
#####----------------------------

#_______________________________________________________________________________
##### Problem 4 of 6/2/2007
#####-------------------------
# The file Pb4.txt reports the number Y (expressed in thousands of units)
# of vehicles registered annually in three countries of the European Union
# (France, Germany and Italy) during a reference period of 10 years.
# Recent economic models describe the behavior of this variable according
# the model:
# Y | (X = x, G = g) = beta0.g + beta1.g * x^2 + eps
# with eps ~ N (0, sigma ^ 2), x = 1, 2,. . . , 10 (year) and
# g = France, Germany, Italy (EU country).
# (a) With the method of least squares, estimate the 7 parameters of the model.
# (b) using appropriate statistical tests, state if you deem necessary to
#     include into the model:
#     1. the variable x^2;
#     2. the variable G;
#     3. the effect of the variable G onto the coefficient that multiplies the
#        regressor x^2;
#     4. the effect of the variable G on the intercept.
# (c) Once identified the "best model", build three prediction intervals
#     for the number of vehicles registered in the three countries 
#     during the eleventh year, so that the three new observations
#     will fall simultaneously within the respective ranges with 95%
#     of probability.

pb4  <- read.table('Pb4.txt')
pb4

x11()
matplot(pb4, type='l',lwd=2, xlab='anno', ylab='autoveicoli')

dev.off()

### question (a)

# We first build the design matrix and the vector of the responses
Anno <- rep(1:10,3)
Anno

Imm <- c(pb4[,1], pb4[,2], pb4[,3])
Imm

# Model: Imm = beta0.g + beta1.g*Anno^2 + eps (Anno=Year)
# con g=0,1,2 [Italy,France,Germany]; E[eps]=0, Var(eps)=sigma^2

# We need to build appropriate dummies to account for the levels of
# the categorical variable G=Italy/France/Germany (3 levels)
# g groups => g-1 dummies (3 groups => 2 dummies)

dFr <- rep(c(1,0), c(10,20))      # dFr = 1 if France,  0 otherwise
dGer<- rep(c(0,1,0), c(10,10,10)) # dGer= 1 if Germany, 0 otherwise
dFr
dGer

# Equivalent Model:
# Imm = b.0 + b.1*dFr + b.2*dGer + b.3*Anno^2 + b.4*dFr*Anno^2 + b.5*dGer*Anno^2

# Indeed:
# beta0.It=b.0;      beta1.It=b.3;
# beta0.Fr=b.0+b.1;  beta1.Fr=b.3+b.4;
# beta0.Ger=b.0+b.2; beta1.Ger=b.3+b.5

dati <- data.frame(Imm   = Imm,
                   Anno2 = Anno^2,
                   dFr   = rep(c(1,0), c(10,20)),      # dummy that selects France
                   dGer  = rep(c(0,1,0), c(10,10,10))) # dummy that selects Germany
dati

fit <- lm(Imm ~ dFr + dGer + Anno2 + Anno2:dFr + Anno2:dGer, data=dati)

# Equivalent syntax:
# fit <- lm(Imm ~ dFr + dGer + I(Anno^2) + I(Anno^2*dFr) + I(Anno^2*dGer),  data=data.frame(Imm = Imm,
#           Anno = Anno, dFr   = rep(c(1,0), c(10,20)), dGer  = rep(c(0,1,0), c(10,10,10))))

summary(fit)

### question (b)
shapiro.test(residuals(fit))
x11()
par(mfrow=c(2,2))
plot(fit)

dev.off()

# 1. the variable x^2;
linearHypothesis(fit,
                 rbind(c(0,0,0,1,0,0),
                       c(0,0,0,0,1,0),
                       c(0,0,0,0,0,1)),
                 c(0,0,0))

# 2. the variable G;
linearHypothesis(fit,
                 rbind(c(0,1,0,0,0,0),
                       c(0,0,1,0,0,0),
                       c(0,0,0,0,1,0),
                       c(0,0,0,0,0,1)),
                 c(0,0,0,0))

#     3. the effect of the variable G onto the coefficient that multiplies the
#        regressor x^2;
linearHypothesis(fit,
                 rbind(c(0,0,0,0,1,0),
                       c(0,0,0,0,0,1)),
                 c(0,0))

#     4. the effect of the variable G on the intercept.
linearHypothesis(fit,
                 rbind(c(0,1,0,0,0,0),
                       c(0,0,1,0,0,0)),
                 c(0,0))

### question (c)
fit2 <- lm(Imm ~ Anno2 + Anno2:dFr + Anno2:dGer, data=dati)
summary(fit2)

nuovi <- data.frame(Anno2 = c(11,11,11)^2, dFr=c(1,0,0), dGer=c(0,1,0))
IP <- predict(fit2, newdata=nuovi, interval='prediction', level=1-0.05/3)
rownames(IP) <- c('Fr','Ger','It')
IP

#_______________________________________________________________________________
##### Problem 4 of 29/6/2011
#####------------------------
# The file People.txt records the tons of waste collected monthly
# in the city of Santander since January 2009 (t = 1) until May 2011
# (t = 29). Assuming a model of the type:
#   Waste = A + B* t  + C * (1-cos (2pi / 12 * t)) + eps
# with eps ~ N(0, sigma^2) and identifying the contribution of the residents
# with the first two factors, and of the tourists with the third addendum, 
# answer the following questions.
# a) Estimate the model parameters
# b) On the basis of the model (a), is there statistical evidence of an increase
#    attributable to residents?
# c) On the basis of the model (a), is there statistical evidence of a significant
#    contribution by tourists?
# d) The University of Cantabria considered that the GROWTH attributable to residents
#    is quantifiable in an increase of 10 quintals per month. Can deny
#    this statement?
# e) Based on the test (b), (c) and (d) propose a possible reduced model and/or
#    constrained and estimate its parameters.
# f) On the basis of the model (e), provide three point forecasts for the waste 
#    that will be collected in June 2011, for waste that will be collected in June
#    2011 due to residents and that which will be collected in June 2011 due to
#    the tourists.

people <- read.table('people.txt', header=T)
people

attach(people)

### question a)
fit <- lm(rifiuti ~ mese + I(1 - cos(2*pi/12*mese)))
summary(fit)

### question b)
shapiro.test(residuals(fit))

x11()
par(mfrow=c(2,2))
plot(fit)

dev.off()

# Test: H0: beta_1==0 vs beta_1!=0
summary(fit)

## or
linearHypothesis(fit,rbind(c(0,1,0)),0)

### question c)
# Test: H0: beta_2==0 vs beta_2!=0
summary(fit)

## or
linearHypothesis(fit,rbind(c(0,0,1)),0)

### question d)
linearHypothesis(fit,rbind(c(0,1,0)),10)

# or (from the summary)
summary(fit)
t <- (coef(fit)[2]-10)/sqrt(diag(vcov(fit))[2])
t
pval <- 2*(1-pt(t,29-(2+1)))
pval

### question e)
rifiuti.vinc <- rifiuti - 10*mese

fit2 <- lm(rifiuti.vinc ~ I(1 - cos(2*pi/12*mese)))
summary(fit2)

### question f)
shapiro.test(residuals(fit2))

x11()
par(mfrow=c(2,2))
plot(fit2)

dev.off()

coefficients(fit2)
C <- rbind(c(1,(1 - cos(2*pi/12*30))),   # total waste in June [mese=30]
           c(1,0),                       # waste due to residents in June 
           c(0,(1 - cos(2*pi/12*30))))   # waste due to tourists in June 
C

pred <- C %*% coefficients(fit2) + c(10*30, 10*30, 0)  # pred=C%*%beta.hat[fit.mod.constrained] + 10*mese[constrained part]
pred

x11()
plot(people, xlim=c(1,30), ylim=c(900,1400))
lines(mese, fitted(fit))
lines(mese, fitted(fit2) + 10*mese, col='blue')
points(c(30,30,30), pred, pch=16)
legend('bottomright',c('Model 1', 'Constrained model'), lty=1, col=c('black','blue'))

graphics.off()
