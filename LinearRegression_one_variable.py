import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#input
x = np.arange(0,100,10)
#output
y = x**2 +3

X = np.array([np.ones(len(x)), x]).T
Y = (y[:, np.newaxis]) #np.newaxis is used to increase the dimesnion of a given vector by one

#Applying the normal equation
theta = inv(X.T.dot(X)).dot(X.T).dot(Y)

#prediction
y_pred = theta[0]+theta[1]*x

plt.scatter(x, y, color='black')
plt.plot(x, y_pred, color='blue', linewidth = 3)
plt.show()

