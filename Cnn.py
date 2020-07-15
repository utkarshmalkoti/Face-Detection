import keras
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPool2D,BatchNormalization,Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical 
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
import matplotlib.pyplot as plt
import pickle
import numpy as np
X_train = np.array(pickle.load(open("Xtrain.pickel","rb")))
X_train = X_train/255
Y_train = np.array(pickle.load(open("Ytrain.pickel","rb")))

X_test = np.array(pickle.load(open("Xtest.pickel","rb")))
X_test = X_test/255
Y_test = np.array(pickle.load(open("Ytest.pickel","rb")))

# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,shuffle=False)

Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(128,(2,2),input_shape=(100,100,1)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
print("set1")

# model.add(Conv2D(128,(2,2)))
# model.add(Dropout(0.3))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2,2)))
# print("set2")

# model.add(Conv2D(100,(3,3)))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
print("set3")

model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
print("set5")

model.add(Dense(4,activation = "softmax"))
print("set last")
sgd = keras.optimizers.SGD(lr=0.05, momentum=0.9,clipvalue = 0.5)
model.compile(optimizer=sgd,loss= "categorical_crossentropy",metrics=['accuracy'])
# rlrop = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5)

checkpoint = ModelCheckpoint("Best_model.hdf5",monitor = 'val_accuracy',save_best_only = True, mode = 'max')

h = model.fit(X_train,Y_train,batch_size=32,epochs=100,validation_data=(X_test,Y_test))
n = np.arange(100)
plt.style.use("ggplot")
plt.plot(n,h.history["loss"],label="Train Loss")
plt.plot(n,h.history["accuracy"],label="Train accuracy")
plt.plot(n,h.history["val_loss"],label="Test Loss")
plt.plot(n,h.history["val_accuracy"],label="Test accuracy")
plt.xlabel("EPOCHES")
plt.ylabel("Acc/Loss")
plt.legend(loc="upper left")
plt.show()