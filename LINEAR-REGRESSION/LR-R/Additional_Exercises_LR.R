library(MASS)
library(car)

#_______________________________________________________________________________
##### Problem 4 of 6/2/2007
#####-------------------------
# The file Pb4.txt reports the number Y (expressed in thousands of units)
# of vehicles registered annually in three countries of the European Union
# (France, Germany and Italy) during a reference period of 10 years.
# Recent economic models describe the behavior of this variable according
# the model:s
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
##### Problema 2 of 28/2/2007
#####--------------------------------------
# Pb2.txt dataset shows average monthly temperature (° C) recorded in
# 2006 in three Canadian locations: Edmonton, Montreal and Resolute.
# It is common in meteorology to assume that the average monthly 
# temperatures fluctuatesinusoidally around an annual average value:
# Temp.g (t) =  beta0.g +beta1.g * sin (2pi / 12 * t) + 
#               beta2.g * cos (2pi / 12 * t) + eps
# with eps ~ N (0,2), t = 1, 2, 3,. . . , 12 (month) and g = Edmonton, 
# Resolute, Montreal (Location).
# (a) Using the least squares method is estimate the 10 parameters of the model
# (b) Verify the model assumptions.
# (c) Taking advantage of the known trigonometric relation
#     sin(alpha-beta) = sin(alpha) * cos(beta) - cos(alpha) * sin(beta)
#     and reinterpreting the model of the form:
#     Tempg (t) = μ.g + A.g * sin (2pi / 12 * (t-phi.g) + eps
#     report the analytical relation between the new parameters
#     (Μ.g, A.g, phi.g) and the old parameters (beta0.g, beta1.g, beta2.g).
# (d) Estimate the parameters of the new formulation, namely:
#     - The annual average values (μ.g).
#     - The oscillation amplitudes (A.g).
#     - The phases of the oscillations (phi'g).
# (e) Through the use of an appropriate statistical test (report the
#     corresponding p-value) justify the possibility of using a reduced model
#     that assume that the fluctuations have same amplitude and phase, but 
#     different annual means in the stations of in Edmonton and Montreal.
Temperature <- read.table('Pb2.txt')
Temperature

x11()
matplot(Temperature, type='l', lwd=2)
dev.off()

temp <- cbind(Tem = c(Temperature[,1],Temperature[,2],Temperature[,3]),
              Sin = sin(2*pi/12*c(1:12,1:12,1:12)),
              Cos = cos(2*pi/12*c(1:12,1:12,1:12)),
              Res = c(rep(0,12), rep(1,12), rep(0,12)), # dummy for Resolute
              Mon = c(rep(0,12), rep(0,12), rep(1,12))) # dummy for Montreal

temp <- data.frame(temp)
temp


### question (a)

fit <- lm(Tem ~ Res + Mon + Sin + Cos + Res*Sin + Res*Cos +Mon*Sin + Mon*Cos, data=temp)
summary(fit)

xplot <- rep(seq(1,12,len=100),3)
nuovi <- data.frame(Sin = sin(2*pi/12*xplot), Cos = cos(2*pi/12*xplot),
                    Res = c(rep(0,100), rep(1,100), rep(0,100)),
                    Mon = c(rep(0,200), rep(1,100)))

x11()
plot(xplot, predict(fit, newdata = nuovi) ,col=rep(1:3,each=100), pch=16)
points(rep(1:12,3), temp$Tem, col='blue', lwd=2)

dev.off()

coef(fit)
Beta <- rbind(
  Edmonton = coef(fit)[c(1,4,5)],
  Resolute = coef(fit)[c(1,4,5)] +  coef(fit)[c(2,6,7)],
  Montreal = coef(fit)[c(1,4,5)] +  coef(fit)[c(3,8,9)])
Beta

### question (b)
shapiro.test(residuals(fit))
shapiro.test(rstudent(fit))

x11()

plot(fitted(fit),residuals(fit))
plot(rep(1:12,3),residuals(fit))

dev.off()

### question (c)
nuovi <- cbind(media = Beta[,1], 
               ampiezza = sqrt(Beta[,2]^2+Beta[,3]^2),
               fase = -12/(2*pi)*atan(Beta[,3]/Beta[,2])+6)

### question (d)
nuovi

### question (e)
linearHypothesis(fit,
                 rbind(c(0,0,0,0,0,0,0,1,0),
                       c(0,0,0,0,0,0,0,0,1)),
                 c(0,0))

#_______________________________________________________________________________
##### Problema 4 of 29/6/2011
#####-------------------------
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

#_______________________________________________________________________________
##### Problem 4 of 1/7/2009
#####-------------------------
# The Index Librorum Prohibitorum (edition of 1948), lists about 10000 works 
# considered heretical by the Catholic Churchlists. The file 'index.txt'
# shows, for the years ranging from 1300 to 1899, the number of works
# added annually to the Index. Most historians believe that the average
# number of works added each year decreased linearly in time during this
# period (model A). Recently, Prof. Langdon proposed a theory according to
# which the linear trend "momentarily" changed (in a discontinuous way) 
# during the French hegemony period (1768, the Treaty of Versailles, 1815, 
# Battle diWaterloo, inclusive), during which a collapse of the works annually 
# added to the Index occurred (Model B). Defining as μ(t) the average number 
# of works added to the Index in year t, and formalizing the two models as 
# follows:
# Model A: μ(t) = alpha + beta * t;
# Model B: μ(t) = alpha1 + beta1 * t for 1768 <= t <= 1815
#          μ(t) = alpha2 + beta2 * t for t <= 1767 or t> = 1816;
# answer the following questions:
# a) estimate the parameters of both models using the method of least squares;
#    which assumptions needs to be introduced in order to get unbiased estimates?
# b) is there statistical evidence of a different linear trend in the period of
#    French hegemony?
# c) using the model (b) and Bonferroni's inequality, they provide 2 90% 
#    global confidence intervals for the mean and the variance of the number
#    of works included in the index in the year 1800;
# d) using the model (b), provide a 90% confidence interval for the difference 
#    between the average number of works added to the Index in 1816 and average 
#    number of works added in 1815.
index <- read.table('index.txt', header=TRUE)
index

rm(Anno); rm(Numero)
attach(index)
x11()
plot(Anno,Numero)

dev.off()

### question a)

# Model A
# Y = alpha + beta*t + eps, E[eps]=0, var(eps)=sigma^2
fitA <- lm(Numero ~ Anno)

summary(fitA)

# Model B
# Y = alpha.g + beta.g*t + eps =
#   = b0 + b1*D + b2*anno + b3*D*anno + eps, E[eps]=0, var(eps)=sigma^2

D <- ifelse(Anno>=1768 & Anno<=1815, 1, 0)

fitB <- lm(Numero ~ D + Anno + D*Anno )
summary(fitB)

alpha <- c(alpha1=coef(fitB)[1]+coef(fitB)[2],alpha2=coef(fitB)[1])
alpha
beta  <- c( beta1=coef(fitB)[3]+coef(fitB)[4], beta2=coef(fitB)[3])
beta

x11()
plot(Anno,Numero)
points(Anno, fitted(fitA), pch=19, col='blue')
points(Anno, fitted(fitB), pch=19)

dev.off()

### question b)

shapiro.test(residuals(fitB))

linearHypothesis(fitB, rbind(c(0,1,0,0),c(0,0,0,1)), c(0,0))

### question c)
k <- 2
alpha <- .1
n <- dim(index)[1]
r <- 3

Z0   <- data.frame(D=1, Anno=1800)
ICBmean <- predict(fitB, Z0, interval='confidence',level=1-alpha/k) 
ICBmean

e <- residuals(fitB)
ICBvar <- data.frame(L=t(e)%*%e/qchisq(1-alpha/(2*k),n-(r+1)),
                     U=t(e)%*%e/qchisq(alpha/(2*k),n-(r+1)))
ICBvar

### question d)
a <- c(0,-1,1,-1815)
Bf <- c('1816-1815_L'= t(a) %*% coefficients(fitB) - sqrt(t(a) %*% vcov(fitB) %*% a) * qt(1 - alpha/2, n-(r+1)),
        '1816-1815_U'= t(a) %*% coefficients(fitB) + sqrt(t(a) %*% vcov(fitB) %*% a) * qt(1 - alpha/2, n-(r+1)) )
Bf

detach(index)

#_______________________________________________________________________________
##### Problema 4 del 4/7/2007
#####-------------------------
# At the Tenaris steel mills, the relationship between length [m] and
# Temperature [° C] of some steel bars that will be sold to Pirelli
# is under study (the data are contained in tenaris.txt file). The relation
# is hypothesized of the kind:
#   L = L0 + C* T + D  * T ^ 2 + eps
# with L the length of the bar, T the temperature of the bar, L0 the length 
# of the bar at 0 °C, C the coefficientof of linear thermal expansion, D
# the coefficient of quadratic thermal expansion and eps a measurement error
# of zero mean.
# Answer the following questions using appropriate statistical arguments:
# a) Estimate the parameters L0, C, D and the variance of error eps.
# b) Based on the analysis of residuals, do you think that there are the
#    conditions to make inference on the coefficients based on a Gaussian
#    model? (In case of Yes proceed to step (c); in case of negative answer
#    identify the problem, remove it and return to point (a))
# c) Do you think that the model explains the possible dependence between 
#    the temperature T and the length L?
# d) do you deem plausible to consider that the length of the bars at 0 °C
#    is equal to 2?
# E) do you think that you can eliminate from the model the quadratic term?

ten <- read.table('tenaris.txt', header=TRUE)
ten

attach(ten)

### question a)

fit <- lm(L ~ T + I(T^2))

summary(fit)

e <- residuals(fit)
S2 <- t(e)%*%e / (df.residual(fit))
S2

### question b)

shapiro.test(e)

x11()
par(mfrow=c(2,2))
plot(fit)

x11()
plot(T,L)
points(ten[1,1],ten[1,2],pch=19)

graphics.off()

detach(ten)

# Remove the outlier
ten1 <- ten[-1,]

fit <- lm(L ~ T + I(T^2), data=ten1)
summary(fit)

e <- residuals(fit)
S2 <- t(e)%*%e / (df.residual(fit))
S2

shapiro.test(e)

x11()
par(mfrow=c(2,2))
plot(fit)

dev.off()

### question c)

attach(ten1)
x11()
plot(T,L)
points(T,fitted(fit),col='blue', pch=19)

dev.off()

summary(fit)$r.squared

### question d)

linearHypothesis(fit, c(1,0,0), 2)

### question e)
summary(fit)

# or
linearHypothesis(fit, c(0,0,1), 0)

#_______________________________________________________________________________
