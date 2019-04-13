### TOPICS:
### LDA and QDA

#_______________________________________________________________________________
### Linear and Quadratic Discriminant Analysis
###
### Example 1 (2 classes, univariate)
###----------------------------------
cito <- read.table('cytokines.txt', header=T)
cito

attach(cito)

A <- which(group=='A')   # Group A: favorable clinical outcome (women for whom 
#           the treatment has effect)
B <- which(group=='B')   # Group B: unfavorable clinical outcome (women for whom 
#           the treatment has no effect)
x11()
plot(cito[,1], cito[,2], pch=19, col=c(rep('blue',8),rep('red',5)), xlab='Inf-g', ylab='IL-5')

dev.off()

### Idea: we aim to find a "rule" to classify patiens as Group A
### or B, given the measurmente of Inf-g and IL-5
### In fact, we only consider the variable Infg since there isn't statistical 
### evidence to state that there is a difference in mean along the component
### IL5 
### -> Exercise: build a confidence region of level 95% for the difference
###    of the means between the two group (independent populations)
###    and verify the previous statement

### LDA (univariate) 
###------------------
# Assumptions:
# 1) if L=i, X.i ~ N(mu.i, sigma.i^2), i=A,B
# 2) sigma.A=sigma.B
# 3) c(A|B)=c(B|A) (equal misclassification costs)

# verify assumptions 1) e 2): 
# 1) normality (univariate) within the groups
shapiro.test(cito[A,1])
shapiro.test(cito[B,1])

# 2) equal variance (univariate)
bartlett.test(Infg ~ group)

nA <- length(A)
nB <- length(B)
n  <- nA + nB

# Prior probabilities
PA <- nA / n
PB <- nB / n

### Recall: the classification region is obtained by comparing pA*f.A and pB*f.B
MA <- mean(Infg[A])
MB <- mean(Infg[B])
SA <- var(Infg[A])
SB <- var(Infg[B])
S  <- ((nA-1) * SA + (nB-1) * SB) / (nA + nB - 2)

x <- seq(-10, 35, 0.5) # include the range of Infg

x11(width = 5)
par(mfrow=c(2,1))

plot(  x, PA * dnorm(x, MA, sqrt(S)), type='l', col='blue', ylab=expression(paste('estimated ', pi[i] * f[i](x))), main='LDA')
points(x, PB * dnorm(x, MB, sqrt(S)), type='l', col='red')
points(Infg[A], rep(0, length(A)), pch=16, col='blue')
points(Infg[B], rep(0, length(B)), pch=16, col='red')
legend(-10, 0.03, legend=c(expression(paste('P(A)',f[A],'(x)')), expression(paste('P(B)',f[B],'(x)'))), col=c('blue','red'), lty=1, cex = 0.7)

plot(  x, PA * dnorm(x, MA, sqrt(S)) / (PA * dnorm(x, MA, sqrt(S)) + PB * dnorm(x, MB, sqrt(S))), type='l', col='blue', ylab='estimated posterior')
points(x, PB * dnorm(x, MB, sqrt(S)) / (PA * dnorm(x, MA, sqrt(S)) + PB * dnorm(x, MB, sqrt(S))), type='l', col='red')
points(Infg[A], rep(0, length(A)), pch=16, col='blue')
points(Infg[B], rep(0, length(B)), pch=16, col='red')
legend(-10, 0.9, legend=c('P(A|X=x)', 'P(B|X=x)'), col=c('blue','red'), lty=1, cex = 0.7)
dev.off()
### end Recall

### LDA with R
library(MASS)
cito.lda <- lda(group ~ Infg)
cito.lda
# Note: if we don't specify the prior, their are estimated
# from the sample

# posterior probability and classification for x=0
x <- data.frame(Infg = 0)
# The command predict() returns a list containing (see the help)
# - the class associated with the 
#   highest probability
predict(cito.lda, x)$class
# - the posterior for the classes
predict(cito.lda, x)$posterior
# - in lda: the coordinates of the canonical analysis of Fisher
predict(cito.lda, x)$x

x11()
plot(Infg[A], rep(0, length(A)), pch=16, col='blue', ylim=c(0,1),
     xlab='x', ylab='estimated posterior', main="LDA", xlim=range(Infg))
points(Infg[B], rep(0, length(B)), pch=16, col='red')
abline(v=0, col='grey')
points(c(0,0),c(predict(cito.lda, data.frame(Infg = 0))$posterior),
       col=c('blue', 'red'), pch='*', cex=2.5)

# posterior probability for a grid of x's
x <- data.frame(Infg=seq(-10, 35, 0.5))

cito.LDA.A <- predict(cito.lda, x)$posterior[,1] # posterior probability for class A
cito.LDA.B <- predict(cito.lda, x)$posterior[,2] # posterior probability for class B

predict(cito.lda, x)$class 
head(predict(cito.lda, x)$posterior)

lines(x[,1], cito.LDA.A, type='l', col='blue', xlab='x', ylab='estimated posterior', main="LDA")
points(x[,1], cito.LDA.B, type='l', col='red')
abline(h = 0.5)
legend(-10, 0.9, legend=c('P(A|X=x)', 'P(B|X=x)'), fill=c('blue','red'), cex = 0.7)

dev.off()


# set prior probabilities
cito.lda.1 <- lda(group ~ Infg, prior=c(0.05,0.95))
cito.lda.1

x <- data.frame(Infg=seq(-10, 35, 0.5))

cito.LDA.A.1 <- predict(cito.lda.1, x)$posterior[,1] # posterior probability for class A
cito.LDA.B.1 <- predict(cito.lda.1, x)$posterior[,2] # posterior probability for class B

x11()
plot  (x[,1], cito.LDA.A.1, type='l', col='blue', xlab='x', ylab='estimated posterior', main="LDA")
points(x[,1], cito.LDA.B.1, type='l', col='red')
abline(h = 0.5)
legend(-10, 0.9, legend=c('P(A|X=x)', 'P(B|X=x)'), fill=c('blue','red'), cex = 0.7)
points(Infg[A], rep(0, length(A)), pch=16, col='blue')
points(Infg[B], rep(0, length(B)), pch=16, col='red')

points(x[,1], cito.LDA.A, type='l', col='grey')
points(x[,1], cito.LDA.B, type='l', col='grey')



### k-nearest neighbor classifier
###-------------------------------
# Let's consider a non-parametric classifier
library(class)
cito.knn <- knn(train = Infg, test = x, cl = group, k = 3, prob=T)
cito.knn.class <- (cito.knn == 'B')+0 
cito.knn.B <- ifelse(cito.knn.class==1, 
                     attributes(cito.knn)$prob, 
                     1 - attributes(cito.knn)$prob)

x11()
plot(x[,1], cito.LDA.B, type='l', col='red', lty=2, xlab='x', ylab='estimated posterior')
points(x[,1], cito.knn.B, type='l', col='red', lty=4)
abline(h = 0.5)
legend(-10, 0.75, legend=c('LDA','knn'), lty=c(2,4), col='red')

# let's change k
x11(width = 28, height = 21)
par(mfrow=c(3,4))
for(k in 1:12)
{
  cito.knn <- knn(train = Infg, test = x, cl = group, k = k, prob=T)
  cito.knn.class <- (cito.knn == 'B')+0 
  cito.knn.B <- ifelse(cito.knn.class==1, attributes(cito.knn)$prob, 1 - attributes(cito.knn)$prob)
  
  plot(x[,1], cito.LDA.B, type='l', col='red', lty=2, xlab='x', ylab='estimated posterior', main=k)
  points(x[,1], cito.knn.B, type='l', col='black', lty=1, lwd=2)
  abline(h = 0.5)
}

detach(cito)

#_______________________________________________________________________________
### Example 2 (3 classes, bivariate)
###-------------------------------------------

###---------------------------------------------------------------------
### We consider only the first two variables, Sepal.Length and Sepal.Width
### (p=2, g=3); we aim to build a classifier that based on the characteristic 
### of the sepal identifies the iris species

attach(iris)

species.name <- factor(Species, labels=c('setosa','versicolor','virginica'))

g=3 

i1 <- which(species.name=='setosa')
i2 <- which(species.name=='versicolor')
i3 <- which(species.name=='virginica')

n1 <- length(i1)
n2 <- length(i2)
n3 <- length(i3)
n <- n1+n2+n3

detach(iris)

iris2 <- iris[,1:2]

# Jittering
set.seed(280787)
iris2 <- iris2 + cbind(rnorm(150, sd=0.025))    # jittering

# plot the data
x11()

plot(iris2, main='Iris Sepal', xlab='Sepal.Length', ylab='Sepal.Width', pch=19)
points(iris2[i1,], col='red', pch=19)
points(iris2[i2,], col='green', pch=19)
points(iris2[i3,], col='blue', pch=19)
legend(min(iris[,1]), max(iris[,2]), legend=levels(species.name), fill=c('red','green','blue'))

dev.off()

m <-  colMeans(iris2)
m1 <- colMeans(iris2[i1,])
m2 <- colMeans(iris2[i2,])
m3 <- colMeans(iris2[i3,])

S1 <- cov(iris2[i1,])
S2 <- cov(iris2[i2,])
S3 <- cov(iris2[i3,])
Sp  <- ((n1-1)*S1+(n2-1)*S2+(n3-1)*S3)/(n-g)

# One-way MANOVA (See LAB 9)
fit <- manova(as.matrix(iris2) ~ species.name)
summary.manova(fit,test="Wilks")

# Linear Discriminant Analysis (LDA)
lda.iris <- lda(iris2, species.name)
lda.iris

# "coefficients of linear discriminants" and "proportion of trace":
# Fisher discriminant analysis. 
# In particular:
# - coefficients of linear discriminants: versors of the canonical directions
#   [to be read column-wise]
# - proportion of trace: proportion of variance explained by the corresponding 
#   canonical direction
names(lda.iris)

Lda.iris <- predict(lda.iris, iris2)
#Lda.iris
names(Lda.iris)

# Compute the APER
Lda.iris$class   # assigned classes
species.name     # true labels
table(class.true=species.name, class.assigned=Lda.iris$class)

errori <- (Lda.iris$class != species.name)
errori
sum(errori)
length(species.name)

APER   <- sum(errori)/length(species.name)
APER

(1+14+14)/150

# Remark: this is correct only if we estimate the prior with the empirical  
#         frequences! Otherwise:
# prior <- c(1/3,1/3,1/3)
# G <- 3
# misc <- table(classe.vera=species.name, classe.allocata=Lda.iris$class)
# APER <- 0
# for(g in 1:G)
# APER <- APER + sum(misc[g,-g])/sum(misc[g,]) * prior[g]  

# Compute the estimate of the AER by cross-validation 
LdaCV.iris <- lda(iris2, species.name, CV=TRUE)  # specify the argument CV

LdaCV.iris$class
species.name
table(classe.vera=species.name, classe.allocataCV=LdaCV.iris$class)

erroriCV <- (LdaCV.iris$class != species.name)
erroriCV
sum(erroriCV)

AERCV   <- sum(erroriCV)/length(species.name)
AERCV
# Remark: correct only if we estimate the priors through the sample frequencies!

# Plot the partition induced by LDA

x11()
plot(iris2, main='Iris Sepal', xlab='Sepal.Length', ylab='Sepal.Width', pch=20)
points(iris2[i1,], col='red', pch=20)
points(iris2[i2,], col='green', pch=20)
points(iris2[i3,], col='blue', pch=20)
legend(min(iris[,1]), max(iris[,2]), legend=levels(species.name), fill=c('red','green','blue'), cex=.7)

points(lda.iris$means, pch=4,col=c('red','green','blue') , lwd=2, cex=1.5)

x  <- seq(min(iris[,1]), max(iris[,1]), length=200)
y  <- seq(min(iris[,2]), max(iris[,2]), length=200)
xy <- expand.grid(Sepal.Length=x, Sepal.Width=y)

z  <- predict(lda.iris, xy)$post  # these are P_i*f_i(x,y)  
z1 <- z[,1] - pmax(z[,2], z[,3])  # P_1*f_1(x,y)-max{P_j*f_j(x,y)}  
z2 <- z[,2] - pmax(z[,1], z[,3])  # P_2*f_2(x,y)-max{P_j*f_j(x,y)}    
z3 <- z[,3] - pmax(z[,1], z[,2])  # P_3*f_3(x,y)-max{P_j*f_j(x,y)}

# Plot the contour line of level (levels=0) of z1, z2, z3: 
# P_i*f_i(x,y)-max{P_j*f_j(x,y)}=0 i.e., boundary between R.i and R.j 
# where j realizes the max.
contour(x, y, matrix(z1, 200), levels=0, drawlabels=F, add=T)  
contour(x, y, matrix(z2, 200), levels=0, drawlabels=F, add=T)
contour(x, y, matrix(z3, 200), levels=0, drawlabels=F, add=T)

library(rgl)
library(mvtnorm)
open3d()
points3d(iris2[i1,1], iris2[i1,2], 0, col='red', pch=15)
points3d(iris2[i2,1], iris2[i3,2], 0, col='green', pch=15)
points3d(iris2[i3,1], iris2[i2,2], 0, col='blue', pch=15)
surface3d(x,y,matrix(dmvnorm(xy, m1, Sp) / 3, 50), alpha=0.4, color='red')
surface3d(x,y,matrix(dmvnorm(xy, m2, Sp) / 3, 50), alpha=0.4, color='green', add=T)
surface3d(x,y,matrix(dmvnorm(xy, m3, Sp) / 3, 50), alpha=0.4, color='blue', add=T)
box3d()


### Quadratic Discriminand Analysis (QDA)
###---------------------------------------

qda.iris <- qda(iris2, species.name)
qda.iris
Qda.iris <- predict(qda.iris, iris2)
#Qda.iris

# compute the APER
Qda.iris$class
species.name
table(classe.vera=species.name, classe.allocata=Qda.iris$class)

erroriq <- (Qda.iris$class != species.name)
erroriq

APERq   <- sum(erroriq)/length(species.name)
APERq
# Remark: correct only if we estimate the priors through the sample frequencies!

# Compute the estimate of the AER by cross-validation 
QdaCV.iris <- qda(iris2, species.name, CV=T)
QdaCV.iris$class
species.name
table(classe.vera=species.name, classe.allocataCV=QdaCV.iris$class)

erroriqCV <- (QdaCV.iris$class != species.name)
erroriqCV

AERqCV   <- sum(erroriqCV)/length(species.name)
AERqCV
# Remark: correct only if we estimate the priors through the sample frequencies!

# Plot the partition induced by QDA
x11()
plot(iris2, main='Iris Sepal', xlab='Sepal.Length', ylab='Sepal.Width', pch=20)
points(iris2[i1,], col='red', pch=20)
points(iris2[i2,], col='green', pch=20)
points(iris2[i3,], col='blue', pch=20)
legend(min(iris[,1]), max(iris[,2]), legend=levels(species.name), fill=c('red','green','blue'))

points(qda.iris$means, col=c('red','green','blue'), pch=4, lwd=2, cex=1.5)

x  <- seq(min(iris[,1]), max(iris[,1]), length=200)
y  <- seq(min(iris[,2]), max(iris[,2]), length=200)
xy <- expand.grid(Sepal.Length=x, Sepal.Width=y)

z  <- predict(qda.iris, xy)$post    
z1 <- z[,1] - pmax(z[,2], z[,3])    
z2 <- z[,2] - pmax(z[,1], z[,3])    
z3 <- z[,3] - pmax(z[,1], z[,2])

contour(x, y, matrix(z1, 200), levels=0, drawlabels=F, add=T)
contour(x, y, matrix(z2, 200), levels=0, drawlabels=F, add=T)
contour(x, y, matrix(z3, 200), levels=0, drawlabels=F, add=T)

open3d()
points3d(iris2[i1,1], iris2[i1,2], 0, col='red', pch=15)
points3d(iris2[i2,1], iris2[i3,2], 0, col='green', pch=15)
points3d(iris2[i3,1], iris2[i2,2], 0, col='blue', pch=15)
surface3d(x,y,matrix(dmvnorm(xy, m1, S1) / 3, 50), alpha=0.4, color='red')
surface3d(x,y,matrix(dmvnorm(xy, m2, S2) / 3, 50), alpha=0.4, color='green', add=T)
surface3d(x,y,matrix(dmvnorm(xy, m3, S3) / 3, 50), alpha=0.4, color='blue', add=T)
box3d()


### knn-classifier
###----------------
# Plot the partition induced by knn

k <- 3

plot(iris2, main='Iris.Sepal', xlab='Sepal.Length', ylab='Sepal.Width', pch=15)
points(iris2[i1,], col=2, pch=15)
points(iris2[i3,], col=4, pch=15)
points(iris2[i2,], col=3, pch=15)
legend(min(iris[,1]), max(iris[,2]), legend=levels(species.name), fill=c(2,3,4))

x  <- seq(min(iris[,1]), max(iris[,1]), length=200)
y  <- seq(min(iris[,2]), max(iris[,2]), length=200)
xy <- expand.grid(Sepal.Length=x, Sepal.Width=y)

iris.knn <- knn(train = iris2, test = xy, cl = iris$Species, k = k)

z  <- as.numeric(iris.knn)

contour(x, y, matrix(z, 200), levels=c(1.5, 2.5), drawlabels=F, add=T)

dev.off()
dev.off()
dev.off()

#_______________________________________________________________________________
##### Problem 2 of 9/09/2009
#####--------------------------
# The vending machines of the Exxon fuel contain optical detectors able
# to measure the size of the banknotes inserted. Knowing that 0.1% of the 
# $ 10 banknotes in circulation are counterfeit, Exxon would like to implement a
# software to identify false $ 10 bills, as to minimize the economic losses. 
# Assuming that:
#  • both the populations of real and false banknotes follow a normal 
#    distribution (with different mean and covariance matrices);
#  • accepting a false banknote leads to an economic loss of $ 10;
#  • rejecting a true banknote brings a economic loss quantifiable in 5 cents;
# satisfy the following requests of the Exxon:
# a) build an appropriate classifier, estimating the unknown parameters
#    starting from the two datasets moneytrue.txt and moneyfalse.txt, containing
#    data about 100 true banknotes and 100 counterfeit banknotes (in mm). 
#    Qualitatively shows two classification regions in a graph;
# b) calculate the APER the classifier and, based on the APER, estimate the 
#    expected economic damage of the classifier.
# c) what is the estimated probabilities that the first 10$ bill inserted in the
#    machine is rejected 

true <- read.table('moneytrue.txt',header=TRUE)
false <- read.table('moneyfalse.txt',header=TRUE)

vf <- factor(rep(c('true','false'),each=100), levels=c('true','false'))

# question a)
mcshapiro.test(true)
mcshapiro.test(false)

c.vf <- 10
c.fv <- 0.05

prior <- c(1-0.001,0.001)
pv <- prior[1]
pf <- prior[2]

# Prior modified to account for the misclassification costs
prior.c <- c(pv*c.fv/(c.vf*pf+c.fv*pv),pf*c.vf/(c.vf*pf+c.fv*pv))
prior.c

# QDA
banconote <- rbind(true,false)
qda.m <- qda(banconote, vf, prior=prior.c)
qda.m

x11()
plot(banconote[,1:2], main='Banknotes', xlab='V1', ylab='V2', pch=20)
points(false, col='red', pch=20)
points(true, col='blue', pch=20)
legend('bottomleft', legend=levels(vf), fill=c('blue','red'), cex=.7)

points(qda.m$means, pch=4,col=c('red','blue') , lwd=2, cex=1.5)

x  <- seq(min(banconote[,1]), max(banconote[,1]), length=200)
y  <- seq(min(banconote[,2]), max(banconote[,2]), length=200)
xy <- expand.grid(V1=x, V2=y)

z  <- predict(qda.m, xy)$post  
z1 <- z[,1] - z[,2] 
z2 <- z[,2] - z[,1]  

contour(x, y, matrix(z1, 200), levels=0, drawlabels=F, add=T)  
contour(x, y, matrix(z2, 200), levels=0, drawlabels=F, add=T)

# question b)

# APER
Qda.m <- predict(qda.m)
table(classe.vera=vf, classe.allocata=Qda.m$class)

APER  <- 2/100*prior[1]+80/100*prior[2]
APER

# Expected economic loss:
80/100*pf*c.vf+2/100*pv*c.fv

# question c)
pv*2/100+pf*20/100

#_______________________________________________________________________________
##### Problem 2 of 1/07/2009
#####--------------------------
# An art historian requires your help to identify a criterion of classification 
# to discriminate the sculptures created by Gian Lorenzo Bernini from those of 
# other contemporary sculptors, based on the weight [Pounds] and height [m] 
# of 100 sculptures of undoubted attribution (Sculptures.txt files). Taking into
# account that Bernini's sculptures are about 25% of the sculptures which have 
# to be classified and that the purpose of the historian is to minimize the expected
# number of misclassifications:
# a) build two classifiers C1 and C2, respectively, assuming for C1 that the data
#    come from two normal populations with equal covariance matrix, and for C2 that
#    the data come from two normal populations with different covariance matrix;
# b) estimate by cross-validation the AER of the two classifiers and comment their
#    values;
# c) how will be classified by the two classifiers a 2 meter high and 4 tons heavy
#    statue?

sculpt <- read.table('sculptures.txt', header=T)
head(sculpt)

autore <- factor(sculpt[,3], levels=c('Bernini', 'Altro'))
autore

bernini <- sculpt[1:50,1:2]
altro <- sculpt[50:100,1:2]

# question a)
mcshapiro.test(bernini)
mcshapiro.test(altro)

# LDA
lda.s <- lda(sculpt[,1:2], autore, prior=c(0.25, 0.75))
lda.s

# QDA
qda.s <- qda(sculpt[,1:2], autore, prior=c(0.25, 0.75))
qda.s

x11()
plot(sculpt[,1:2], main='Sculptures', xlab='Height', ylab='Weight', pch=20)
points(bernini, col='red', pch=20)
points(altro, col='blue', pch=20)
legend('bottomleft', legend=levels(autore), fill=c('red','blue'), cex=.7)

points(lda.s$means, pch=4,col=c('red','blue') , lwd=2, cex=1.5)

x  <- seq(min(sculpt[,1]), max(sculpt[,1]), length=200)
y  <- seq(min(sculpt[,2]), max(sculpt[,2]), length=200)
xy <- expand.grid(Altezza=x, Peso=y)

z  <- predict(lda.s, xy)$post  
z1 <- z[,1] - z[,2] 
z2 <- z[,2] - z[,1]  

z.q  <- predict(qda.s, xy)$post  
z1.q <- z.q[,1] - z.q[,2] 
z2.q <- z.q[,2] - z.q[,1]  

contour(x, y, matrix(z1, 200), levels=0, drawlabels=F, add=T)  
contour(x, y, matrix(z2, 200), levels=0, drawlabels=F, add=T)

contour(x, y, matrix(z1.q, 200), levels=0, drawlabels=F, add=T, lty=2)  
contour(x, y, matrix(z2.q, 200), levels=0, drawlabels=F, add=T, lty=2)

dev.off()

# question b)
# LDA
LdaCV.s <- lda(sculpt[,1:2], autore, prior=c(0.25, 0.75), CV=T)
table(classe.vera=autore, classe.allocataCV=LdaCV.s$class)

AER.CV.l <- 45/50*0.25+3/50*0.75
AER.CV.l

# QDA
QdaCV.s <- qda(sculpt[,1:2], autore, prior=c(0.25, 0.75), CV=T)
table(classe.vera=autore, classe.allocataCV=QdaCV.s$class)

AER.CV.q <- 20/50*0.25+6/50*0.75
AER.CV.q

# question c)
predict(lda.s, c(Altezza=2, Peso=4))
predict(qda.s, c(Altezza=2, Peso=4))

points(2,4, pch=3, col='springgreen', lwd=2)

graphics.off()

Qda.m <- predict(qda.s)
table(classe.vera=autore, classe.allocata=Qda.m$class)

###-------------------------------------------------------------------------
### FISHER DISCRIMINANT ANALYSIS
###-----------------------------------
### Let's change viewpoint: we look for the directions that highlight
### the discrimination among groups
### -> we look for the canonical directions

# Remark. Assumptions: homogeneity of the covariance structure
# [we relax the normal assumption]

# Let's consider again the iris dataset
attach(iris)

species.name <- factor(Species, labels=c('setosa','versicolor','virginica'))

g=3 

i1 <- which(species.name=='setosa')
i2 <- which(species.name=='versicolor')
i3 <- which(species.name=='virginica')

n1 <- length(i1)
n2 <- length(i2)
n3 <- length(i3)
n <- n1+n2+n3

detach(iris)

iris2 <- iris[,1:2]
set.seed(280787)
iris2 <- iris2 + cbind(rnorm(150, sd=0.025))    # jittering

m <-  colMeans(iris2)
m1 <- colMeans(iris2[i1,])
m2 <- colMeans(iris2[i2,])
m3 <- colMeans(iris2[i3,])

S1 <- cov(iris2[i1,])
S2 <- cov(iris2[i2,])
S3 <- cov(iris2[i3,])
Sp  <- ((n1-1)*S1+(n2-1)*S2+(n3-1)*S3)/(n-g)

# covariance among groups (estimate)
B <- 1/n*(n1* cbind(m1 - m) %*% rbind(m1 - m) +
            n2* cbind(m2 - m) %*% rbind(m2 - m) +
            n3* cbind(m3 - m) %*% rbind(m3 - m))
B

# covariance within groups (estimate)
Sp

# how many coordinates?
g <- 3
p <- 2
s <- min(g-1,p)
s

# Matrix Sp^(-1/2)
val.Sp <- eigen(Sp)$val
vec.Sp <- eigen(Sp)$vec
invSp.2 <- 1/sqrt(val.Sp[1])*vec.Sp[,1]%*%t(vec.Sp[,1]) + 1/sqrt(val.Sp[2])*vec.Sp[,2]%*%t(vec.Sp[,2])
invSp.2 

spec.dec <- eigen(invSp.2 %*% B %*% invSp.2)

# first canonical coordinate
a1 <- invSp.2 %*% spec.dec$vec[,1]
a1

cc1.iris <- as.matrix(iris2)%*%a1

# second canonical coordinate
a2 <- invSp.2 %*% spec.dec$vec[,2]
a2

cc2.iris <- as.matrix(iris2)%*%a2

# compare with the output of lda():
lda.iris
a1
a2
spec.dec$val/sum(spec.dec$val)

### How are the data classified?
# Compute the canonical coordinates of the data
coord.cc=cbind(as.matrix(iris2)%*%cbind(lda.iris$scaling[,1]),
               as.matrix(iris2)%*%cbind(lda.iris$scaling[,2]))
# Compute the coordinates of the mean within groups along the canonical directions
cc.m1 <- c(m1%*%a1, m1%*%a2)
cc.m2 <- c(m2%*%a1, m2%*%a2)
cc.m3 <- c(m3%*%a1, m3%*%a2)
# Assign data to groups
f.class=rep(0, n)
for(i in 1:n) # for each datum
{
  # Compute the Euclidean distance of the i-th datum from mean within the groups
  dist.m=c(d1=sqrt(sum((coord.cc[i,]-cc.m1)^2)),
           d2=sqrt(sum((coord.cc[i,]-cc.m2)^2)),
           d3=sqrt(sum((coord.cc[i,]-cc.m3)^2)))
  # Assign the datum to the group whose mean is the nearest
  f.class[i]=which.min(dist.m)
}
f.class
table(classe.vera=species.name, classe.allocata=f.class)

errors <- (Lda.iris$class != species.name)
sum(errors)
length(species.name)

APERf   <- sum(errors)/length(species.name)
APERf

### How do I classify a new observation?
x.new=c(5.85, 2.90)
# compute the canonical coordinates
cc.new=c(x.new%*%a1, x.new%*%a2)
# compute the distance from the means
dist.m=c(d1=sqrt(sum((cc.new-cc.m1)^2)),
         d2=sqrt(sum((cc.new-cc.m2)^2)),
         d3=sqrt(sum((cc.new-cc.m3)^2)))
# assign to the nearest mean
which.min(dist.m)

color.species=c('red','green','blue')

# visually
x11(width=14, height=7)
par(mfrow=c(1,2))
plot(iris2[,1], iris2[,2], main='Plane of original coordinates', 
     xlab='Sepal.Length', ylab='Sepal.Width', pch=20, col=as.character(color.species))
legend(min(iris[,1]), min(iris[,2])+2, legend=levels(species.name), fill=c('red','green','blue'), cex=.7)
points(x.new[1], x.new[2], col='gold', pch=19)
points(m1[1], m1[2], pch=4,col='red' , lwd=2, cex=1.5)
points(m2[1], m2[2], pch=4,col='green' , lwd=2, cex=1.5)
points(m3[1], m3[2], pch=4,col='blue' , lwd=2, cex=1.5)

plot(cc1.iris, cc2.iris, main='Plane of canonical coordinates', xlab='first canonical coordinate', ylab='second canonical coordinate', pch=20, col=as.character(color.species))
legend(min(cc1.iris), min(cc2.iris)+2, legend=levels(species.name), fill=c('red','green','blue'), cex=.7)

points(cc.m1[1], cc.m1[2], pch=4,col='red' , lwd=2, cex=1.5)
points(cc.m2[1], cc.m2[2], pch=4,col='green' , lwd=2, cex=1.5)
points(cc.m3[1], cc.m3[2], pch=4,col='blue' , lwd=2, cex=1.5)

points(cc.new[1], cc.new[2], col='gold', pch=19)
points(cc.new[1], cc.new[2], col='green', pch=1)

dev.off()

### We plot the partition generated by the canonical coordinates
color.species <- species.name
levels(color.species) <- c('red','green','blue')

x11()
plot(cc1.iris, cc2.iris, main='Fisher discriminant analysis', xlab='first canonical coordinate', ylab='second canonical coordinate', pch=20, col=as.character(color.species))
legend(min(cc1.iris), min(cc2.iris)+2, legend=levels(species.name), fill=c('red','green','blue'), cex=.7)

points(cc.m1[1], cc.m1[2], pch=4,col='red' , lwd=2, cex=1.5)
points(cc.m2[1], cc.m2[2], pch=4,col='green' , lwd=2, cex=1.5)
points(cc.m3[1], cc.m3[2], pch=4,col='blue' , lwd=2, cex=1.5)

x.cc  <- seq(min(cc1.iris),max(cc1.iris),len=200)
y.cc  <- seq(min(cc2.iris),max(cc2.iris),len=200)
xy.cc <- expand.grid(cc1=x.cc, cc2=y.cc)

z  <- cbind( sqrt(rowSums(scale(xy.cc,cc.m1,scale=FALSE)^2)), sqrt(rowSums(scale(xy.cc,cc.m2,scale=FALSE)^2)), sqrt(rowSums(scale(xy.cc,cc.m3,scale=FALSE)^2)))
z1.cc <- z[,1] - pmin(z[,2], z[,3])    
z2.cc <- z[,2] - pmin(z[,1], z[,3])    
z3.cc <- z[,3] - pmin(z[,1], z[,2])

contour(x.cc, y.cc, matrix(z1.cc, 200), levels=0, drawlabels=F, add=T)
contour(x.cc, y.cc, matrix(z2.cc, 200), levels=0, drawlabels=F, add=T)
contour(x.cc, y.cc, matrix(z3.cc, 200), levels=0, drawlabels=F, add=T)

dev.off()

# Plot LDA
x11()
plot(iris2, main='Iris Sepal', xlab='Sepal.Length', ylab='Sepal.Width', pch=20)
points(iris2[i1,], col='red', pch=20)
points(iris2[i2,], col='green', pch=20)
points(iris2[i3,], col='blue', pch=20)
legend(min(iris[,1]), max(iris[,2]), legend=levels(species.name), fill=c('red','green','blue'), cex=.7)

points(lda.iris$means, pch=4,col=c('red','green','blue') , lwd=2, cex=1.5)

x  <- seq(min(iris[,1]), max(iris[,1]), length=200)
y  <- seq(min(iris[,2]), max(iris[,2]), length=200)
xy <- expand.grid(Sepal.Length=x, Sepal.Width=y)

z  <- predict(lda.iris, xy)$post  # these are P_i*f_i(x,y)  
z1 <- z[,1] - pmax(z[,2], z[,3])  # P_1*f_1(x,y)-max{P_j*f_j(x,y)}  
z2 <- z[,2] - pmax(z[,1], z[,3])  # P_2*f_2(x,y)-max{P_j*f_j(x,y)}    
z3 <- z[,3] - pmax(z[,1], z[,2])  # P_3*f_3(x,y)-max{P_j*f_j(x,y)}

# Plot the contour line of level (levels=0) of z1, z2, z3: 
# P_i*f_i(x,y)-max{P_j*f_j(x,y)}=0 i.e., boundary between R.i and R.j 
# where j realizes the max.
contour(x, y, matrix(z1, 200), levels=0, drawlabels=F, add=T)  
contour(x, y, matrix(z2, 200), levels=0, drawlabels=F, add=T)
contour(x, y, matrix(z3, 200), levels=0, drawlabels=F, add=T)

dev.off()

###########################################################################
# Plot of the projections on the canonical directions (non orthogonal!)
x11(width=14, height=7)
par(mfrow=c(1,2))
plot(iris2, main='Projection on the canonical directions', xlab='Sepal.Length', ylab='Sepal.Width', pch=20, xlim=c(-3,8), ylim=c(-3,7))
points(iris2[i1,], col='red', pch=20)
points(iris2[i2,], col='green', pch=20)
points(iris2[i3,], col='blue', pch=20)
legend('topleft', legend=levels(species.name), fill=c('red','green','blue'), cex=.7)

points(lda.iris$means, pch=4,col=c('red','green','blue') , lwd=2, cex=1.5)

abline(coef=c(0,(a1[2]/a1[1])), col='grey55',lty=2)
abline(coef=c(0,(a2[2]/a2[1])), col='grey55',lty=2)

abline(h=0,v=0, col='grey35')

points(cc1.iris*a1[1]/(sum(a1^2)),cc1.iris*a1[2]/(sum(a1^2)),col=as.character(color.species))
points(cc2.iris*a2[1]/(sum(a2^2)),cc2.iris*a2[2]/(sum(a2^2)),col=as.character(color.species))

arrows(x0=0, y0=0, x1=a1[1], y1=a1[2], length=.1)
arrows(x0=0, y0=0, x1=a2[1], y1=a2[2], length=.1)

text(a1[1], a1[2], 'a1',pos=3)
text(a2[1], a2[2], 'a2',pos=2)

plot(cc1.iris, cc2.iris, main='Coordinate system of the canonical coordinates', xlab='first canonical coordinate', ylab='second canonical coordinate', pch=20, col=as.character(color.species))
legend('topleft', legend=levels(species.name), fill=c('red','green','blue'), cex=.7)

cc.m1 <- c(m1%*%a1, m1%*%a2)
cc.m2 <- c(m2%*%a1, m2%*%a2)
cc.m3 <- c(m3%*%a1, m3%*%a2)

points(cc.m1[1], cc.m1[2], pch=4,col='red' , lwd=2, cex=1.5)
points(cc.m2[1], cc.m2[2], pch=4,col='green' , lwd=2, cex=1.5)
points(cc.m3[1], cc.m3[2], pch=4,col='blue' , lwd=2, cex=1.5)

dev.off()
