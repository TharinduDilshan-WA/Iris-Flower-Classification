from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#load data set
iris = load_iris()
X = iris['data']
Y = to_categorical(iris['target'])

#split data to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# create model
model = Sequential()
model.add(Dense(6, input_dim = 4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss ='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=10)
