import hashlib
import dhash 
import os 
from hashlib import md5
from PIL import Image

celeb_path = os.path.join("celebrities")
celebs = os.listdir(celeb_path)

for celeb in celebs:
    duplicates = []
    hashkeys = []
    img_path = os.path.join(celeb_path,celeb)
    face_path = os.path.join(img_path)
    imgs = os.listdir(face_path)

    for img in imgs:
        with open(os.path.join(face_path,img),'rb') as f:
            img_hash = md5(f.read()).hexdigest()
            if img_hash not in hashkeys:
                hashkeys.append(img_hash)
            else:
                print("already there",img)
                duplicates.append(img)
                
    for i in duplicates:
        os.remove(os.path.join(face_path,i))
    print("Duplicates",celeb,":",duplicates)