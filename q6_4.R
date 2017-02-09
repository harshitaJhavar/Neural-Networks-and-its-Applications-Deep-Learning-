#Solution for question 6.4

#Solution of Part a

# load the data
mydata = read.csv("https://www.lsv.uni-saarland.de/fileadmin/teaching/psr/ws1617/logReg.csv", header = TRUE)

# plot the data
plot(mydata$u[mydata$y == 0], mydata$v[mydata$y == 0],, xlab="u", ylab="v")
points(mydata$u[mydata$y == 1], mydata$v[mydata$y == 1], col="blue", pch=3)
legend("topright", c("y=0","y=1"), pch=c(1, 3), col=c("black", "blue"), bty="n")

#Solution for Part b

# sigmoid function
g = function (z) {
  return (1 / (1 + exp(-z) ))
} # plot(g(c(1,2,3,4,5,6)), type="l")

# build hight order feature vector
# for 2 features, for a given degree
hi.features = function (f1,f2,deg) {
  n = ncol(f1)
  ma = matrix(rep(1,length(f1)))
  for (i in 1:deg) {
    for (j in 0:i)    
      ma = cbind(ma, f1^(i-j) * f2^j)
  }
  return(ma)
}


h = function (x,th) {
  return(g(x %*% th))
} # h(x,th)

#Solution to part c)

# derivative of J 
grad = function (x,y,th,m,la) {
  G = (la/m * th)
  G[1,] = 0
  return((1/m * t(x) %*% (h(x,th) - y)) +  G)
} # grad(x,y,th,m,la)

H = function (x,y,th,m,la) {
  n = length(th)
  L = la/m * diag(n)
  L[1,] = 0
  return((1/m * t(x) %*% x * diag(h(x,th)) * diag(1 - h(x,th))) + L)
} # H(x,y,th,m,la)

# cost function
J = function (x,y,th,m,la) {
  pt = th
  pt[1] = 0
  A = (la/(2*m))* t(pt) %*% pt
  return((1/m * sum(-y * log(h(x,th)) - (1 - y) * log(1 - h(x,th)))) + A)
} # J(x,y,th,m,la)

# Solution to part d - Newton's method

# setup variables
m = length(mydata$u) # samples
x = hi.features(mydata$u, mydata$v,6)
n = ncol(x) # features
y = matrix(mydata$y, ncol=1)

# lambda = 1
th1 = matrix(0,n)
la = 1
jiter = array(0,c(15,1))
for (i in 1:15) {
  jiter[i] = J(x,y,th1,m,la)
  th1 = th1 - solve(H(x,y,th1,m,la)) %*% grad(x,y,th1,m,la) 
}

# lambda = 0
th0 = matrix(0,n)
la = 0
for (i in 1:15) {
  th0 = th0 - solve(H(x,y,th0,m,la)) %*% grad(x,y,th0,m,la) 
}

# lambda = 10
th10 = matrix(0,n)
la = 10
for (i in 1:15) {
  th10 = th10 - solve(H(x,y,th10,m,la)) %*% grad(x,y,th10,m,la) 
}

#Solution to part e)

# calculate the decision boundary line

u = seq(-1, 1.2, len=200);
v = seq(-1, 1.2, len=200);
z0 = matrix(0, length(u), length(v))
z1 = matrix(0, length(u), length(v))
z10 = matrix(0, length(u), length(v))
for (i in 1:length(u)) {
  for (j in 1:length(v)) {
    z0[j,i] =  hi.features(u[i],v[j],6) %*% th0
    z1[j,i] =  hi.features(u[i],v[j],6) %*% th1
    z10[j,i] =  hi.features(u[i],v[j],6) %*% th10
  }
}

# plots
contour(u,v,z0,nlev = 0, xlab="u", ylab="v", nlevels=0, col="black",lty=2)
contour(u,v,z1,nlev = 0, xlab="u", ylab="v", nlevels=0, col="red",lty=2, add=TRUE)
contour(u,v,z10,nlev = 0, xlab="u", ylab="v", nlevels=0, col="green3",lty=2, add=TRUE)
points(mydata$u[mydata$y == 0], mydata$v[mydata$y == 0])
points(mydata$u[mydata$y == 1], mydata$v[mydata$y == 1], col="blue", pch=3)
legend("topright",  c(expression(lambda==0), expression(lambda==1),expression(lambda==10)), lty=1, col=c("black", "red","green3"),bty="n" )