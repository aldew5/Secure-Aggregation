from train import *
from model import *
import time, datetime
import pandas as pd

num_epochs = 100 * 61
batch_size = 48
lr = 1e-4


X_train = pd.read_csv('./data/train_X.csv')
y_train = pd.read_csv('./data/train_y.csv')
X_train = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
y_train = np.array([i[0] for i in y_train.values])


X_test = pd.read_csv('./data/test_X.csv',header=None)
y_test = pd.read_csv('./data/test_y.csv',header=None)
X_test = np.append(X_test, np.ones((len(X_test), 1)), axis=1)
y_test = np.array([i[0] for i in y_test.values])

start = datetime.datetime.now()
LR = LinearRegression(lr, num_epochs, batch_size, None)
LR.train(X_train, y_train)
mse, rsquare = LR.eval(X_test, y_test)
print("START TIME:", start)
print("FINISH TIME:", datetime.datetime.now())
print("MSE:", mse)
print("R^2:", rsquare)
