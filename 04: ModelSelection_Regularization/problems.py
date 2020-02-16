import numpy as np
import matplotlib.pyplot as plt
# import scipy
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
# import tensorflow as tf
# import keras


# The holdout method    
# (answer1: model complexity 3 is the best)
# (answer2: )
def holdoutMethod():
# a)
    data = np.load("04_model_selection_data.npy")   # [predictor, target] = [x, y]
    x = data[:, 0]
    y = data[:, 1]
    # Split into train and test Sets
    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, test_size=0.2, random_state=10)
    x_train = x_train.reshape([x_train.shape[0],1]) # Transpose x_train
    x_test = x_test.reshape([x_test.shape[0],1])    # Transpose x_test       
# b)
    meanSquaredErrors_Train = []
    meanSquaredErrors_Test = []
    bias_Train = []
    variance_Test = []
    for n in range(1,7):
    # Preprocessing (get X matrix) -> (X = features matrix)
        polyFeatures = PolynomialFeatures(n)                # tells how many features there are (depending on polynomial order)
        x_polyTrain = polyFeatures.fit_transform(x_train)   # by transform means to reshape data only
        x_polyTest = polyFeatures.fit_transform(x_test)     # by transform means to reshape data only        
    # Fit Model
        polyModel = LinearRegression()
        polyModel.fit(x_polyTrain, y_train)               
        predictionTrain = polyModel.predict(x_polyTrain)
        predictionTest = polyModel.predict(x_polyTest)        
    # Graph Model        
        x_plot_train, y_plot_train = zip(*sorted(zip(x_train,predictionTrain)))
        x_plot_test, y_plot_test = zip(*sorted(zip(x_test,predictionTest)))
        realX_train, realY_train = zip(*sorted(zip(x_train,y_train)))
        realX_test, realY_test = zip(*sorted(zip(x_test,y_test)))        
        plt.subplot(2,3,n)
        plt.plot(x_plot_train, y_plot_train, 'r')    
        plt.plot(x_plot_test, y_plot_test, 'g')                         
        plt.plot(realX_train, realY_train, 'ro')         
        plt.plot(realX_test, realY_test, 'go')    
        plt.title("Complexity " + str(n))
    # Mean Squared Error
        MSE_train = mean_squared_error(realY_train, y_plot_train)
        MSE_test = mean_squared_error(realY_test, y_plot_test)
        meanSquaredErrors_Train.append(MSE_train)
        meanSquaredErrors_Test.append(MSE_test)
        # Represent MSE using BIAS and VARIANCE
        # error = E[(f + ε − predF)^2] = bias[predF]^2 + var[y] + var[predF]
        # E[(f + ε − predF)^2] = (f − E[predF])^2 + E[ε^2] + E[(E[predF] − predF)^2]
        # 
        # BIAS = (f − E[predF])^2 
        # MSE_training = (f − E[predF])^2 + E[ε^2]
            # Since we cannot get E[ε^2] in practice (we dont know the exact function),
            # we cancel down E[ε^2], then:
            # BIAS = MSE_training
        # VARIANCE = E[(E[predF] − predF)^2]
        # MSE_test = (f − E[predF])^2 + E[ε^2] + E[(E[predF] − predF)^2]
            # VARIANCE = MSE_training - MSE_training
        bias_Train.append(MSE_train)
        variance_Test.append(MSE_test - MSE_train)
    plt.show()
    complexities = np.arange(start=1, stop=7, step=1)
    plt.bar(complexities-0.4, meanSquaredErrors_Train, width=0.4, align='center', color='r')
    plt.bar(complexities, meanSquaredErrors_Test, width=0.4, align='center', color='g')
    plt.title("MSE vs Complexity")
    plt.show()
    plt.bar(complexities-0.4, bias_Train, width=0.4, align='center', color='r')
    plt.bar(complexities, variance_Test, width=0.4, align='center', color='g')
    plt.title("BIAS,VAR vs Complexity")
    plt.show()    
# holdoutMethod()

# k-fold cross-validation
# (answer: k=20 is the most stable pattern)
def kFold_crossValidation():
# a)
    data = np.load("04_model_selection_data.npy")   # [predictor, target] = [x, y]
    x = data[:, 0]
    y = data[:, 1]
    kList = [2, 4, 5, 10, 20]
    for k in kList:
        # Split into train and test Sets
        kf = KFold(k, shuffle=True, random_state=10)
        for trainIndexes, testIndexes in kf.split(x):
            x_train, x_test = x[trainIndexes], x[testIndexes]
            y_train, y_test = y[trainIndexes], y[testIndexes]
        x_train = x_train.reshape([x_train.shape[0],1]) # Transpose x_train
        x_test = x_test.reshape([x_test.shape[0],1])    # Transpose x_test      
# b)
        meanSquaredErrors_Train = []
        meanSquaredErrors_Test = []
        for n in range(1,7):
        # Preprocessing (get X matrix) -> (X = features matrix)
            polyFeatures = PolynomialFeatures(n)                # tells how many features there are (depending on polynomial order)
            x_polyTrain = polyFeatures.fit_transform(x_train)   # by transform means to reshape data only
            x_polyTest = polyFeatures.fit_transform(x_test)     # by transform means to reshape data only        
        # Fit Model
            polyModel = LinearRegression()
            polyModel.fit(x_polyTrain, y_train)            
            predictionTrain = polyModel.predict(x_polyTrain)
            predictionTest = polyModel.predict(x_polyTest)        
        # Graph Model        
            x_plot_train, y_plot_train = zip(*sorted(zip(x_train,predictionTrain)))
            x_plot_test, y_plot_test = zip(*sorted(zip(x_test,predictionTest)))
            realX_train, realY_train = zip(*sorted(zip(x_train,y_train)))
            realX_test, realY_test = zip(*sorted(zip(x_test,y_test)))        
            plt.subplot(2,3,n)
            plt.plot(x_plot_train, y_plot_train, 'r')    
            plt.plot(x_plot_test, y_plot_test, 'g')                         
            plt.plot(realX_train, realY_train, 'ro')         
            plt.plot(realX_test, realY_test, 'go')    
            plt.title("Complexity " + str(n))
        # Mean Squared Error
            MSE_train = mean_squared_error(realY_train, y_plot_train)
            MSE_test = mean_squared_error(realY_test, y_plot_test)
            meanSquaredErrors_Train.append(MSE_train)
            meanSquaredErrors_Test.append(MSE_test)
        plt.show()
        complexities = np.arange(start=1, stop=7, step=1)
        plt.bar(complexities-0.4, meanSquaredErrors_Train, width=0.4, align='center', color='r')
        plt.bar(complexities, meanSquaredErrors_Test, width=0.4, align='center', color='g')
        plt.title("MSE vs Complexity")
        plt.show()
# kFold_crossValidation()


# Regularization (using k=20 and modelComplexityGrade=3)
def regularization():
    data = np.load("04_model_selection_data.npy")   # [predictor, target] = [x, y]
    x = data[:, 0]
    y = data[:, 1]
# a)
    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, test_size=0.2, random_state=10)
    x_train = x_train.reshape([x_train.shape[0],1]) # Transpose x_train
    x_test = x_test.reshape([x_test.shape[0],1])    # Transpose x_test      
# Preprocessing (get X matrix) -> (X = features matrix)
    polyFeatures = PolynomialFeatures(3)                # tells how many features there are (depending on polynomial order)
    x_polyTrain = polyFeatures.fit_transform(x_train)   # by transform means to reshape data only
    x_polyTest = polyFeatures.fit_transform(x_test)     # by transform means to reshape data only        
# Fit Model
    polyModel = LinearRegression()
    polyModel.fit(x_polyTrain, y_train)                 # train model, (y_train[0] = predictedY)
    predictionTrain = polyModel.predict(x_polyTrain)
    predictionTest = polyModel.predict(x_polyTest)        
# Graph Model        
    x_plot_train, y_plot_train = zip(*sorted(zip(x_train,predictionTrain)))
    x_plot_test, y_plot_test = zip(*sorted(zip(x_test,predictionTest)))
    realX_train, realY_train = zip(*sorted(zip(x_train,y_train)))
    realX_test, realY_test = zip(*sorted(zip(x_test,y_test)))       
    lampdas = np.arange(0,31, 3)
    for lampda in lampdas:
        lampda /= 10000     # λ ∈ [0, 0.003]
        plt.subplot(1,3,1)
        plt.plot(x_plot_train, y_plot_train, 'r')    
        plt.plot(x_plot_test, y_plot_test, 'g')                         
        plt.plot(realX_train, realY_train, 'ro')         
        plt.plot(realX_test, realY_test, 'go')    
        plt.title("PolyRegression 3")
    # Mean Squared Error
        MSE_train = mean_squared_error(realY_train, y_plot_train)
        MSE_test = mean_squared_error(realY_test, y_plot_test)
        # print(MSE_train, MSE_test)
    # b) Ridge Regression
        ridge = Ridge(random_state=10, tol=lampda)
        ridge.coef_ = 9
        ridge.fit(x_polyTrain, y_train)            
        ridgeTrain = ridge.predict(x_polyTrain)
        ridgeTest = ridge.predict(x_polyTest)       
        plt.subplot(1,3,2)
        x_plot_trainRidge, y_plot_trainRidge = zip(*sorted(zip(x_train,ridgeTrain)))
        x_plot_testRidge, y_plot_testRidge = zip(*sorted(zip(x_test,ridgeTest)))     
        plt.plot(x_plot_trainRidge, y_plot_trainRidge, 'r')    
        plt.plot(x_plot_testRidge, y_plot_testRidge, 'g')                         
        plt.plot(realX_train, realY_train, 'ro')         
        plt.plot(realX_test, realY_test, 'go')    
        plt.title("Rigde 9, lampda="+str(lampda))
    # c) Lasso Regression
        lasso = Lasso(random_state=10, tol=lampda)
        lasso.coef_ = 9
        lasso.fit(x_polyTrain, y_train)            
        lassoTrain = lasso.predict(x_polyTrain)
        lassoTest = lasso.predict(x_polyTest)       
        plt.subplot(1,3,3)
        x_plot_trainLasso, y_plot_trainLasso = zip(*sorted(zip(x_train,lassoTrain)))
        x_plot_testLasso, y_plot_testLasso = zip(*sorted(zip(x_test,lassoTest)))     
        plt.plot(x_plot_trainLasso, y_plot_trainLasso, 'r')    
        plt.plot(x_plot_testLasso, y_plot_testLasso, 'g')                         
        plt.plot(realX_train, realY_train, 'ro')         
        plt.plot(realX_test, realY_test, 'go')    
        plt.title("Lasso 9, lampda="+str(lampda))
        plt.show()        
regularization()