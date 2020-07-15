<h1> Face Detection and Face Recognition with Keras</h1>
<hr>
<h2> Goal</h2>
<hr>
Scrape images from Goolge Images, Faces Detection using Haar Cascade, Face Recognition

<h2> Installations </h2>
- Install virtual env.
- Install Webdriver in the folder.
- Install Selenium.
- Install PIL.
- Install OpenCV
- Install Numpy
- Install Tensorflow and keras.

'''sh
pip install virtualenv
pip install selenium
pip install pillow
pip install opencv-python
pip install tensorflow
pip install keras
'''

<h2> Scrapping images from Google Images</h2>
- Open *scraping.py*.
- Run python script.
- Enter the list of Celebrities and to end the list enter -1.

This Python script will search 4 celebrities at a time and scrape them parallelly to save time and save all of the images
in a folder with respective celebrity name.
<hr>
<h2> Image Preprocessing </h2>
- Open *hashing.py* and run.
This program is made to remove duplicate images.

<h2> Face Detection</h2>
- Open python folder and copy "data" folder and face it inside your folder
- Open *Face_detect.py* and run.
This Python script will detect faces from dowloaded images, crop them and save in a folder named *faces* inside respective celebrity folder
with 100*100 resolution.
-> Now, run *hashing.py* just to be sure that there are no duplicate images.<br>
-> Then, delete all the unwanted images or bad quality images from the *faces* folder.

<h2> Data Augmentation </h2>
- Open and run *data_aug.py*.
 This python script will increase the data set by changing some specifications of the current image dataset.

<h2>Dataset</h2>
-Open and run *dataset.py*
This will convert images into numpy array and save them in a pickle file as 'x.pickle' and 'y.pickle'. Where 'x.pickle' contains numpy arrays of all the images
and 'y.pickle' contains labels.

<h2>Face Recognition Model</h2>
- Open and run *face_rec.py*
This CNN model will split the dataset int training and testing dataset and train the model.