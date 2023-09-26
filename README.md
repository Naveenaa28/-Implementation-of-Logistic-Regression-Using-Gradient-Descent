# Implementation-of-Logistic-Regression-Using-Gradient-Descent
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
 ## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:212221220035
RegisterNumber: Naveenaa V.R 
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X = data[:, [0,1]]
y = data[:, 2]

print("Array value of X:")
X[:5]

print("Array value of Y:")
y[:5]

print("Exam 1-score graph:")
plt.figure()
plt.scatter(X[y==1][:, 0],X[y==1][:, 1], label="Admitted")
plt.scatter(X[y==0][:, 0],X[y==0][:, 1], label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print("Sigmoid function graph:")
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return J,grad
  
print("X_train_grad value:")
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

print("Y_train_grad value:")
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  return J
  
def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y) / X.shape[0]
  return grad 
  
print("Print res.x:")
X_train = np.hstack((np.ones((X.shape[0], 1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y), method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max= X[:,0].min()-1, X[:,0].max()+1
  y_min, y_max= X[:,0].min()-1, X[:,0].max()+1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted") 
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
print("Decision boundary-graph for exam score:")
plotDecisionBoundary(res.x,X,y)

print("Probability value:")
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)
  
print("Prediction value of mean:")
np.mean(predict(res.x,X) == y)
```

## Output:
![image](https://github.com/Naveenaa28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131433133/140a347e-c22e-4530-8be4-996b542b93c0)
![image](https://github.com/Naveenaa28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131433133/1cdbfaef-a799-467e-80e8-b4f3cdeccb26)
![image](https://github.com/Naveenaa28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131433133/20c6058a-266c-4081-8ac3-7822d09a872f)
![image](https://github.com/Naveenaa28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131433133/ea5c9394-1c00-4248-b9e5-cba74a19c593)
![image](https://github.com/Naveenaa28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131433133/4591c53c-f155-411e-a0b6-0eb3031163a5)
![image](https://github.com/Naveenaa28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131433133/da24b581-190a-4f35-b9c8-8c5f702d34ca)
![image](https://github.com/Naveenaa28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131433133/b29ab4fb-38f4-4210-8279-16612385a031)
![image](https://github.com/Naveenaa28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131433133/415aeefe-c9c1-49e0-be30-7a68af39cb1d)
![image](https://github.com/Naveenaa28/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/131433133/2a6ccf27-24ac-4584-bdc2-5c796dce9f8a)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

