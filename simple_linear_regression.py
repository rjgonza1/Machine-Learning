
# coding: utf-8

# # Homework 2
# ### Ryan Gonzales
# ### 57555019

# In[291]:


import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

curve = np.genfromtxt("data/curve80.txt", delimiter=None)
scalar = curve[:,0] # first column is scalar
X = np.atleast_2d(scalar).T
Y = curve[:,1]
Xtr, Xte, Ytr, Yte = ml.splitData(X, Y, .75)


# ## Problem 1
# ### Number 1

# In[11]:


print("Printing Shapes: \n")
print("Xtr:", Xtr.shape)
print("Xte:", Xte.shape)
print("Ytr:", Ytr.shape)
print("Yte:", Yte.shape)


# ### Number 2 

# In[204]:


lr = ml.linear.linearRegress(Xtr, Ytr)
xs = np.linspace(0,10,200)
xs = xs[:,np.newaxis]
ys = lr.predict(xs)

plt.scatter(Xtr, Ytr, color=['k','b'])
plt.plot(xs, ys)


# In[149]:


print("Regression coefficients:", lr.theta)

# Verfies the x intercept
assert(ys[0,0] == lr.theta[0,0]) 

# Verifies the slope, check two points just to be sure
assert ys[1,0] == lr.theta[0,1] * xs[1,0] + ys[0,0]
assert ys[2,0] == lr.theta[0,1] * xs[2,0] + ys[0,0]


# In[38]:


print("MSE Training data:", lr.mse(Xtr, Ytr))
print("MSE Test data:", lr.mse(Xte, Yte))


# ###  Number 3 

# In[223]:


# Part a

# MSE arrays for part b
mse_tr = [0 for i in range(6)]
mse_te = [0 for i in range(6)]

mse_tr[0] = lr.mse(Xtr, Ytr)
mse_te[0] = lr.mse(Xte, Yte)


# ys_3 is degree 3
XtrP,params = ml.transforms.rescale(ml.transforms.fpoly(Xtr, 3, bias=False))
XteP,_= ml.transforms.rescale(ml.transforms.fpoly(Xte,3,False), params)
XsP,_ = ml.transforms.rescale(ml.transforms.fpoly(xs,3,False), params)

lr_3 = ml.linear.linearRegress(XtrP, Ytr)
ys_3 = lr_3.predict(XsP)

mse_tr[1] = lr_3.mse(XtrP, Ytr)
mse_te[1] = lr_3.mse(XteP, Yte)

# ys_5 is degree 5
XtrP,params = ml.transforms.rescale(ml.transforms.fpoly(Xtr, 5, bias=False))
XteP,_= ml.transforms.rescale(ml.transforms.fpoly(Xte,5,False), params)
XsP,_ = ml.transforms.rescale(ml.transforms.fpoly(xs,5,False), params)

lr_5 = ml.linear.linearRegress(XtrP, Ytr)
ys_5 = lr_5.predict(XsP)

mse_tr[2] = lr_5.mse(XtrP, Ytr)
mse_te[2] = lr_5.mse(XteP, Yte)

# ys_7 is degree 7
XtrP,params = ml.transforms.rescale(ml.transforms.fpoly(Xtr, 7, bias=False))
XteP,_= ml.transforms.rescale(ml.transforms.fpoly(Xte,7,False), params)
XsP,_ = ml.transforms.rescale(ml.transforms.fpoly(xs,7,False), params)

lr_7 = ml.linear.linearRegress(XtrP, Ytr)
ys_7 = lr_7.predict(XsP)

mse_tr[3] = lr_7.mse(XtrP, Ytr)
mse_te[3] = lr_7.mse(XteP, Yte)

# ys_10 is degree 10
XtrP,params = ml.transforms.rescale(ml.transforms.fpoly(Xtr, 10, bias=False))
XteP,_= ml.transforms.rescale(ml.transforms.fpoly(Xte,10,False), params)
XsP,_ = ml.transforms.rescale(ml.transforms.fpoly(xs,10,False), params)

lr_10 = ml.linear.linearRegress(XtrP, Ytr)
ys_10 = lr_10.predict(XsP)

mse_tr[4] = lr_10.mse(XtrP, Ytr)
mse_te[4] = lr_10.mse(XteP, Yte)

# ys_18 is degree 18
XtrP,params = ml.transforms.rescale(ml.transforms.fpoly(Xtr, 18, bias=False))
XteP,_= ml.transforms.rescale(ml.transforms.fpoly(Xte,18,False), params)
XsP,_ = ml.transforms.rescale(ml.transforms.fpoly(xs,18,False), params)

lr_18 = ml.linear.linearRegress(XtrP, Ytr)
ys_18 = lr_18.predict(XsP)

mse_tr[5] = lr_18.mse(XtrP, Ytr)
mse_te[5] = lr_18.mse(XteP, Yte)

# Plots
# Degree = 1
fig1, ax = plt.subplots(1, 1, figsize=(10,8))
ax.plot(xs, ys)
ax.scatter(Xtr, Ytr, color=['k','b'])
ax.set_ylim(-3,9)
ax.set_title("Degree = 1")


# Degree = 3
fig2, ax = plt.subplots(1, 1, figsize=(10,8))
ax.plot(xs, ys_3)
ax.scatter(Xtr, Ytr, color=['k','b'])
ax.set_ylim(-3,9)
ax.set_title("Degree = 3")

# Degree = 5
fig3, ax = plt.subplots(1, 1, figsize=(10,8))
ax.plot(xs, ys_5)
ax.scatter(Xtr, Ytr, color=['k','b'])
ax.set_ylim(-3,9)
ax.set_title("Degree = 5")

# Degree = 7
fig4, ax = plt.subplots(1, 1, figsize=(10,8))
ax.plot(xs, ys_7)
ax.scatter(Xtr, Ytr, color=['k','b'])
ax.set_ylim(-3,9)
ax.set_title("Degree = 7")

# Degree = 10
fig5, ax = plt.subplots(1, 1, figsize=(10,8))
ax.plot(xs, ys_10)
ax.scatter(Xtr, Ytr, color=['k','b'])
ax.set_ylim(-3,9)
ax.set_title("Degree = 10")

# Degree = 18
fig6, ax = plt.subplots(1, 1, figsize=(10,8))
ax.plot(xs, ys_18)
ax.scatter(Xtr, Ytr, color=['k','b'])
ax.set_ylim(-3,9)
ax.set_title("Degree = 18")

plt.show()


# In[225]:


# Part b
plt.semilogy(mse_tr, 'r', mse_te, 'g')

#The x axis corresponds to the array index
#In this case [0, 1, 2, 3, 4, 5] correspond to [1, 3, 5, 7, 10, 18]


# Part c
# 
# Based on the plot from part b, there is the least error on the test data at index 4, which is degree 10. 10 fit the data closest, but 18 fit it too much, and thus overfitting occurred.

# ## Problem 2 

# In[332]:


nFolds = 5
degrees = [1,3,5,7,10,18]
mse_fold = [0 for i in range(6)]
J = [0 for i in range(nFolds)]
    
def cross_validate(degree):
    XtrP,params = ml.transforms.rescale(ml.transforms.fpoly(Xtr, degree, bias=False))
    XteP,_= ml.transforms.rescale(ml.transforms.fpoly(Xte,degree,False), params)
    XsP,_ = ml.transforms.rescale(ml.transforms.fpoly(xs,degree,False), params)
    
    for iFold in range(nFolds):
        Xti, Xvi, Yti, Yvi = ml.crossValidate(XtrP, Ytr, nFolds, iFold)
        learner = ml.linear.linearRegress(Xti, Yti)
        J[iFold] = learner.mse(Xvi, Yvi)  


# ### Number 1

# In[333]:


for i, d in enumerate(degrees):    
    cross_validate(d)
    sum = 0
    for e in J:
        sum += e
    mse_fold[i] = sum / len(J)
    
plt.semilogy(mse_fold)


# ### Number 2 

# The MSE from cross-validation is very similar to the MSE from problem 1. There is still the overfitting problem with a model of degree 18, but cross-validation is slightly different (see Number 3).

# ### Number 3

# Contrary to the plot from problem 1, the best degree is at index 2, which is degree of 5. The MSE plot from problem 1 supported degree = 10, but cross-validation supports degree = 5, which I tend to agree with more, since it's more general.

# ### Number 4 

# In[367]:


nFold_arr = [2,3,4,5,6,10,12,15]
cv_err = [0 for i in range(len(nFold_arr))]

def cross_validate_f(index, folds):
    XtrP,params = ml.transforms.rescale(ml.transforms.fpoly(Xtr, 5, bias=False))
    XteP,_= ml.transforms.rescale(ml.transforms.fpoly(Xte,5,False), params)
    XsP,_ = ml.transforms.rescale(ml.transforms.fpoly(xs,5,False), params)
    J2 = [0 for i in range(folds)]
    
    for iFold in range(folds):
        Xti, Xvi, Yti, Yvi = ml.crossValidate(XtrP, Ytr, folds, iFold)
        learner = ml.linear.linearRegress(Xti, Yti)
        J2[iFold] = learner.mse(Xvi, Yvi)     
    sum = 0
    for e in J2:
        sum += e
    cv_err[index] = sum / len(J2)
        
for i, f in enumerate(nFold_arr):
    cross_validate_f(i, f)  
    
plt.semilogy(cv_err)
print(cv_err)


# For lower k-folds, the error jumps around. This is because you are taking the error on a subset of the data, and there aren't enough subsets to give you a consistent answer. Higher k-folds begin to increase in error due to overfitting. Generally, k-folds that are in the middle will give you the lowest MSE on test data.

# ## Problem 3

# As of my submission, I have not collaborated with anyone.
