import numpy as np
import matplotlib.pyplot as plt
# import scipy
# import sklearn
# import tensorflow
# import keras

##############################################################
####################### PROBLEM 1 ############################
##############################################################
def quantile_quantile():
    N = 1000
    x = np.random.rand(int(N/2)) 
    x = np.append(x, np.random.rand(int(N/2))*-1.0)
    y = np.random.rand(int(N/2))
    y = np.append(y, np.random.rand(int(N/2))*-1.0)
    d = np.sqrt((np.square(x) + np.square(y)))
    array, bins = np.histogram(d, 10)
    plt.subplot(1, 3, 1)
    plt.plot(bins, np.append(array, 0))
    plt.title("Sample")

    xN = np.random.randn(int(N/2)) 
    xN = np.append(xN, np.random.randn(int(N/2))*-1.0)
    yN = np.random.randn(int(N/2))
    yN = np.append(yN, np.random.randn(int(N/2))*-1.0)
    dN = np.sqrt((np.square(xN) + np.square(yN)))
    arrayN, binsN = np.histogram(dN, 10)
    plt.subplot(1, 3, 2)
    plt.plot(binsN, np.append(arrayN, 0))
    plt.title("Normal Distribution")

    qs = np.arange(100)/100.0
    qSample = np.quantile(d,qs)
    qNormal = np.quantile(dN,qs)
    plt.subplot(1, 3, 3)
    plt.scatter(qSample,qNormal)
    plt.plot([0,1.2], [0,3],color='b')
    plt.xlabel("Sample Q")
    plt.ylabel("Normal Q")
    plt.title("Q-Q-plot")

    plt.show()

# print(quantile_quantile())
##############################################################
####################### PROBLEM 2 ############################
##############################################################
def isPrimeNumber(x):
    if(x<2):
        return False
    if(x==2):
        return True
    primes = [2]
    for i in range(3, x-1):
        isPrime = True
        for prime in primes:
            if(i%prime == 0): 
                isPrime = False               
                break
        if(isPrime):
            primes.append(i)
    for prime in primes:
        if(x%prime == 0):
            return False
    return True
##############################################################
####################### PROBLEM 3 ############################
##############################################################
def aproximatePI():
    averageRatio = 0.0
    repetitions = 1000
    i=0
    while(i < repetitions):
        N = 1000000
        x = np.random.rand(int(N/2)) 
        x = np.append(x, np.random.rand(int(N/2))*-1.0)
        y = np.random.rand(int(N/2))
        y = np.append(y, np.random.rand(int(N/2))*-1.0)
        d = np.sqrt((np.square(x) + np.square(y)))
        C = np.sum(d <= 1.0)
        averageRatio += C/N
        i += 1
    averageRatio /= repetitions
    radio = 1    
    areaSquare = (radio*2)*(radio*2) # (lado * lado)
    PI = averageRatio*areaSquare / radio
    return PI

print(aproximatePI())