#import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#creates starting set 500 in each set
n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T
X = np.vstack((Xa, Xb))
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

#plt.scatter(X[:n_pts,0], X[:n_pts,1])
#plt.scatter(X[n_pts:,0], X[n_pts:,1])
#plt.show()

#The sequential model is a linear stack of layers
#Documentation: https://keras.io/models/sequential/
model = Sequential()
#creates a 1 layer dense network with 1 node and 2 inputs with a sigmoid activation function
model.add(Dense(units = 1, input_shape = (2,), activation = "sigmoid")) # why (2,) not (2)
#makes use of adam optimizer with learning rate of .1
adam = Adam(lr = 0.1)
#compiles the model making use of adam, determing loss function (cross entropy) and any metrics
#you want returned like accuracy of the model
model.compile(adam, loss = "binary_crossentropy", metrics=["accuracy"])
#trains the model for 50 epochs, verbose determines how you see training process
# the batch size (the number of training examples per epoch) shuffle determines if
#training data is shuffled and x are numpy arrays with training data and y is an array of labels
h = model.fit(x=X, y=y, verbose=1, batch_size=50, epochs=50, shuffle="true")

#displays the accuracy over epoch
plt.plot(h.history["acc"])
plt.title("accuracy")
plt.xlabel("epochs")
plt.legend(['accuracy'])
plt.show()

#displays the loss over epoch
plt.plot(h.history["loss"])
plt.title("loss")
plt.xlabel("epochs")
plt.legend(['loss'])
plt.show()

#creates a grid with given boundaries dividing data after model has been run
def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:, 0])-1, max(X[:, 0])+1, 50)
    y_span = np.linspace(min(X[:, 1])-1, max(X[:, 1])+1, 50)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)

#runs above function and displays it along with a new data point which it catgorizes and predicts
#if it is 1 or 0 with a decimal
plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
x = 10
y = 10
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker="o", markersize=10, color="red")
print("prediction is", prediction)
plt.show()
