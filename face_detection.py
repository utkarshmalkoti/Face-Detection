import cv2.cv2 as cv2
import numpy as np
import os

profile_cascade = cv2.CascadeClassifier('C:/Users/unnat/OneDrive/Desktop/projects/Scrapping/Cascade/data/haarcascade_profileface.xml')
frontal_cascade = cv2.CascadeClassifier('C:/Users/unnat/OneDrive/Desktop/projects/Scrapping/Cascade/data/haarcascade_frontalface_alt2.xml')
frontal_cascade_default = cv2.CascadeClassifier('C:/Users/unnat/OneDrive/Desktop/projects/Scrapping/Cascade/data/haarcascade_frontalface_default.xml')
frontal_cascade_alt = cv2.CascadeClassifier('C:/Users/unnat/OneDrive/Desktop/projects/Scrapping/Cascade/data/haarcascade_frontalface_alt.xml')
frontal_cascade_cat = cv2.CascadeClassifier('C:/Users/unnat/OneDrive/Desktop/projects/Scrapping/Cascade/data/haarcascade_frontalcatface.xml')
frontal_cascade_cat_ext = cv2.CascadeClassifier('C:/Users/unnat/OneDrive/Desktop/projects/Scrapping/Cascade/data/haarcascade_frontalcatface_extended.xml')
frontal_cascade_tree = cv2.CascadeClassifier('C:/Users/unnat/OneDrive/Desktop/projects/Scrapping/Cascade/data/haarcascade_frontalface_alt_tree.xml')
celebs = os.listdir('Celebrities')
errors = []
for celeb in celebs:
    all_imgs_path = os.path.join('Celebrities',celeb)
    all_imgs = os.listdir(all_imgs_path)
    for img in all_imgs:
        try:
            color_img = cv2.imread(os.path.join(all_imgs_path,img),1)
            gray_img= cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
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
                print(x,y,w,h,"---",img)
                face = color_img[y:y+w+10,x:x+h+5]
                face_folder_path = os.path.join(all_imgs_path,"face")
                ext = all_imgs[0].split('.')[-1]
                if not os.path.exists(face_folder_path):
                    os.makedirs(face_folder_path)
                cv2.imwrite(os.path.join(face_folder_path,f"{img}"),cv2.resize(face,(100,100)))
                break
        except:
            errors.append(img)
            continue
    print(celeb,errors)
    break
cv2.waitKey(0)
