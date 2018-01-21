# Originally run in Jupyter Notebook
# mltools is a library of functions given by my course professor

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

#Parse data
iris = np.genfromtxt("data/iris.txt", delimiter=None)
Y = iris[:,-1]
#first two features
X1 = iris[:,0:2]

####Number 1####
#Randomize data and split into 75/25 train/validation
np.random.seed(0)
X1,Y = ml.shuffleData(X1,Y)

Xtr,Xva,Ytr,Yva = ml.splitData(X1,Y,.75)

#Neighbors array
K = [1,5,10,50]

for i, k in enumerate(K):
    plt.figure(i)
    plt.title("K=" + str(k))
    knn = ml.knn.knnClassify() #Create object and train it
    knn.train(Xtr, Ytr, k)
    ml.plotClassify2D(knn, Xtr, Ytr)  #Visualize data set and decision regions
    
####Number 2####

#New Neighbors array
K = [1,2,5,10,50,100,200]

errTrain = []
errVal = []

for k in K:
    learner = ml.knn.knnClassify(Xtr, Ytr, k)
    errVal.append(learner.err(Xva, Yva))
    errTrain.append(learner.err(Xtr, Ytr))

plt.figure(5)
plt.title("First Two Features Error Rate")
plt.semilogx(errTrain, 'r', errVal, 'g')

# ####Number 3####
Y2 = iris[:,-1]
#all features
X2 = iris[:,0:-1]

np.random.seed(0)
X2,Y2 = ml.shuffleData(X2,Y2)

Xtr2,Xva2,Ytr2,Yva2 = ml.splitData(X2,Y2,.75)

# #Uses same neighbor array K

for j, k in enumerate(K):
    learner = ml.knn.knnClassify(Xtr2, Ytr2, k)
    errVal[j] = learner.err(Xva2, Yva2)
    errTrain[j] = learner.err(Xtr2, Ytr2)

plt.figure(6)
plt.title("All Features Error Rate")
plt.semilogx(errTrain, 'r', errVal, 'g')
