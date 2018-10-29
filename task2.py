import numpy as np
import matplotlib.pyplot as plt
import math


def euler(x,y,h):
    for i in range(1,len(x)):
        y[i]=y[i-1]+ h*f(x[i-1],y[i-1])

def impr_euler(x,y,h):
    for i in range(1,len(x)):
        k1=f(x[i-1],y[i-1])
        k2=f(x[i-1]+h,y[i-1]+h*k1)
        y[i]=y[i-1]+ h/2*(k1+k2)

def runge_kutta(x,y,h):
    for i in range(1,len(x)):
        k1=f(x[i-1],y[i-1])
        k2=f(x[i-1]+h/2,y[i-1]+(h/2)*k1)
        k3=f(x[i-1]+h/2,y[i-1]+(h/2)*k2)
        k4=f(x[i-1] +h,y[i-1]+h*k3)
        y[i]=y[i-1]+(h/6)*(k1+2*k2+2*k3+k4)

def f(x,y):
    return -2*y + 4*x

def real_f(x):
    return 2*x -1 + np.exp(-2*x)

#def show_euler(y1,y2,y3):

def show_errors(y1,y2,y3):
    y6 = [0] * len(x)
    y7 = [0] * len(x)
    y8 = [0] * len(x)
    for i in range(len(y5)):
        y6[i] = y5[i] - y1[i]
        y7[i] = y5[i] - y2[i]
        y8[i] = y5[i] - y3[i]
    plt.plot(x, y6, label="Euler")
    plt.plot(x, y7, label="ImprEuler")
    plt.plot(x, y8, label="Runge")
    plt.legend(loc='upper left')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.ylim(-0.01, 0.01)
    plt.title('Errors')
    plt.show()

x0=0
X=3
n=100
h= (X - x0)/n

x = np.arange(x0,X,h)
y= [0]*len(x)
y[0]=0

y1=[0]*len(x)
y2=[0]*len(x)
y3=[0]*len(x)
y1[0]=y2[0]=y3[0]=0
euler(x,y1,h)
impr_euler(x,y2,h)
runge_kutta(x,y3,h)
#line1 = plt.plot(x, y1, label='Euler')
#line2 = plt.plot(x, y2, label='Imp-Euler')
#line3 = plt.plot(x, y3, label='Runge-Kutta')

x5=np.arange(x0,X,h)
y5=[0]*len(y)
for i in range (len(x5)):
    y5[i] = real_f(x[i])

show_errors(y1,y2,y3)
