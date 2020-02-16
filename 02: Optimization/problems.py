import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import random
# import sklearn
# import tensorflow
# import keras

#######################################
# Analytical minimization of a function
#######################################
# a) -1.4142, 0, 1.4142
# b)
# c) Plot x[-2,2]
def plotL():
    xs=[]
    ys=[]    
    x=-2
    while(x <= 2):
        y = x**4 - 4*x**2 + 4
        xs.append(x)
        ys.append(y)
        x+=0.01
    plt.plot(xs,ys)
    plt.plot([-math.sqrt(2),-math.sqrt(2)], [0,1],color="r")
    plt.plot([math.sqrt(2),math.sqrt(2)], [0,1],color="r")
    plt.plot([0,0], [3,4],color="g")
    plt.show()     
# plotL()

######################################
# Numerical minimization of a function
######################################
# a)
# b)
def evaluateDerivate(actualX):
    y = 4.0*actualX**3 - 8.0*actualX
    return y
def firstGradientDecent():
    LR = 0.01
    actualX = random.randrange(-200,201, 1)/100.0
    for x in range(50):
        actualX = actualX - LR*evaluateDerivate(actualX)
    print(actualX)
def secondGradientDecent():
    LR = np.linspace(0.05,0.15, num=11, endpoint=True)
    actualX = np.full(11, 1.0)
    for x in range(100):
        actualX = actualX - LR*evaluateDerivate(actualX)
    print(actualX)   
    differences = []
    for x in actualX:
        differences.append(abs(math.sqrt(2) - x))
    print(differences)
def thirdGradientDecent():
    N = np.linspace(1,100, num=101, dtype=int, endpoint=True)
    actualX = 1
    LR = 0.01
    estimations = []
    for n in N:
        for x in range(n):
            actualX = actualX - LR*evaluateDerivate(actualX)
        estimations.append(actualX) 
    differences = []
    for x in estimations:
        differences.append(abs(math.sqrt(2) - x))
    print(differences)    
firstGradientDecent()
# secondGradientDecent()
# thirdGradientDecent()


###############################
# The role of the learning rate
###############################
def evaluateV(actualX):
    return actualX**2
def evaluateDerivateV(actualX):
    return 2*actualX
def plotVWithLRs():
    LR = np.linspace(1,11, num=12, dtype=int, endpoint=True)/10.0
    actualX = np.full(12, 1)
    predictedX = []
    predictedY = []
    N = 10
    for x in range(N):    
        actualX = actualX - LR*evaluateDerivateV(actualX)
        predictedX.append(actualX)
        predictedY.append(evaluateV(actualX))
    print(predictedX)
    print(predictedY)

    for graph in range(12):
        xs=[]
        ys=[]    
        x=-2
        while(x <= 2):
            y = x**2
            xs.append(x)
            ys.append(y)
            x+=0.01
        plt.plot(xs,ys)
        print(predictedX[graph])
        print(predictedY[graph])                 
        print('\n')
        plt.plot(predictedX[graph], predictedY[graph], 'ro')
        plt.show()

# plotVWithLRs()


