import keras
from keras.layers import Conv2D,Dense,Dropout,Flatten,BatchNormalization
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import pickle
import numpy as np
import cv2
from PIL import Image
import os

X = np.array(pickle.load(open("Xall.pickle","rb")))
Y = np.array(pickle.load(open("Yall.pickle","rb")))

Y = to_categorical(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

try:
    json_file = open("Cnn_model.json","r").read()
    model = model_from_json(json_file)
    model.load_weights("Best_new_weights.hdf5")
    print("Loaded Weights")
    model.compile(optimizer='sgd',loss="categorical_crossentropy",metrics=['accuracy'])
    loss,acc = model.evaluate(X_test,Y_test,batch_size=5)
    print("Accuracy = ",acc)
    print("Loss= ",loss)
    # print("Loss = ",loss)
except:
    model = Sequential()
    model.add(Conv2D(64,(6,6),activation="relu",strides=(2,2),input_shape=(100,100,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(32,(6,6),activation="relu",strides=(2,2)))

    model.add(Dense(128,activation="relu"))
    model.add(Flatten())
    model.add(Dense(len(np.unique(Y)),activation="softmax"))

    adam = keras.optimizers.Adam(learning_rate=0.05)

    model_json = model.to_json()
    with open("Cnn_model.json", "w") as json_file:
        json_file.write(model_json)

    checkpoint = ModelCheckpoint("Best_new_weights.hdf5",monitor = 'val_accuracy',save_best_only = True, mode = 'max')

    model.compile(optimizer='sgd',loss="categorical_crossentropy",metrics=['accuracy'])
    m = model.fit(X_train,Y_train,batch_size=5,epochs=10,shuffle=True,validation_data=(X_test,Y_test))#,callbacks=[checkpoint])

    n = np.arange(10)
    plt.style.use("ggplot")
    plt.plot(n,m.history["loss"],label="Train Loss")
    plt.plot(n,m.history["accuracy"],label="Train accuracy")
    plt.plot(n,m.history["val_loss"],label="Test Loss")
    plt.plot(n,m.history["val_accuracy"],label="Test accuracy")
    plt.xlabel("EPOCHES")
    plt.ylabel("Acc/Loss")
    plt.legend(loc="upper left")
    plt.show()





def detect_face(color_img,img_add):
    profile_cascade = cv2.CascadeClassifier('Cascade/data/haarcascade_profileface.xml')
    frontal_cascade = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_alt2.xml')
    frontal_cascade_default = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_default.xml')
    frontal_cascade_alt = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_alt.xml')
    frontal_cascade_cat = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalcatface.xml')
    frontal_cascade_cat_ext = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalcatface_extended.xml')
    frontal_cascade_tree = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_alt_tree.xml')
    gray_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
    face = frontal_cascade.detectMultiScale(gray_img,scaleFactor=1.4,minNeighbors=3)
    if face == ():
        face = profile_cascade.detectMultiScale(gray_img,scaleFactor=1.4,minNeighbors=3)
    if face == ():
        face = frontal_cascade_default.detectMultiScale(gray_img,scaleFactor=1.4,minNeighbors=3)
    if face == ():
        face = frontal_cascade_tree.detectMultiScale(gray_img,scaleFactor=1.4,minNeighbors=3)
    if face == ():
        face = frontal_cascade_cat.detectMultiScale(gray_img,scaleFactor=1.4,minNeighbors=3)
    if face == ():
        face = frontal_cascade_cat_ext.detectMultiScale(gray_img,scaleFactor=1.4,minNeighbors=3)
    if face == ():
        face = frontal_cascade_alt.detectMultiScale(gray_img,scaleFactor=1.4,minNeighbors=3)
    if face == ():
        face = frontal_cascade_cat_ext.detectMultiScale(gray_img,scaleFactor=1.4,minNeighbors=3)
    if face == ():
        face = frontal_cascade_tree.detectMultiScale(gray_img,scaleFactor=1.4,minNeighbors=3)

    for (x,y,w,h) in face:
        # print(x,y,w,h,"---",img)
        face = gray_img[y:y+w+10,x:x+h+5]
        resized_face = cv2.resize(face,(100,100))
        resized_face = resized_face.reshape(1,100,100,1)
        break
    return x,y,w,h, resized_face

img_add = input("Enter the complete address of the image: ")
color_img = cv2.imread(img_add,1)
x,y,w,h, resized_face = detect_face(color_img,img_add)
probs = model.predict(np.array(resized_face))
celeb_idx = np.argmax(probs)
celebs = os.listdir("New train celebs")
img = cv2.rectangle(color_img,(x,y),(x+h+5,y+w+10),(0, 0, 255),2)
img = cv2.putText(img,celebs[celeb_idx],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
cv2.imshow(celebs[celeb_idx],img)
cv2.waitKey(0)
print(celebs[celeb_idx])






