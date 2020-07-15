import cv2.cv2 as cv2
from keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img,load_img
import os
img_gen = ImageDataGenerator(rotation_range=50,zoom_range=0.2,horizontal_flip=True,shear_range=0.2,fill_mode='nearest')


celebs = os.listdir("celebrities")
for celeb in celebs:
    img_path = os.path.join(f"celebrities//{celeb}","face")
    all_imgs = os.listdir(img_path)

    for img in all_imgs:
        img_data = load_img(os.path.join(img_path,img))
        img_array = img_to_array(img_data)
        img_array = img_array.reshape(1,100,100,3)
        i=0
        for batch in img_gen.flow(img_array,save_to_dir=img_path,save_prefix=celeb,save_format="jpg",batch_size=1):
            i+=1
            if(i==3):
                break
    


