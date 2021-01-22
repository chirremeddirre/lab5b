#!/usr/bin/env python

#!/usr/bin/env python
from keras.models import load_model
from imutils import paths
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import re
import pandas as pd
import pickle
from keras.utils.data_utils import get_file
from zipfile import ZipFile

#cv2.__version__
# !!NOTE!! Everything was written to comply with openCV 4.1.2 other versions might throw errors
# Accuracy: 96%
#
# TODO: I added exlamation points to clearly mark the lines which you have to modify.
# There is a comment on each of these lines telling you how you should modify it.

# In case you wish to upload your image folder as a zipfile

# Training set uploaded as zip-file



def my_clasif(file_name):
  file_name = 'imagedata.zip' # <-- name of your zipfile

  with ZipFile(file_name, 'r') as zip:
    zip.extractall()
    print('Done')
    print(file_name)




  # Path to true labels (headless csv) Does not need to have the csv extension, just needs to be comma separated.
  #label_path = 'labels.txt' # <- !!!Path to ground truth file goes here!!!

  # Path to image folder
  read_path = file_name.split('.')[0] # <- !!!Name of your folder after unzipping in the previous step!!!

  # !!!You don't have to touch anything below this!!!

  model_name = get_file('captcha_model.hdf5', 'https://github.com/chirremeddirre/lab5b/blob/master/captcha_model.hdf5?raw=true')
  one_hot = get_file('one_hot', 'https://github.com/chirremeddirre/lab5b/blob/master/one_hot?raw=true')
  # Used for printing, not necessary
  # Load true labels
  #true_labels = pd.read_csv(label_path, header=None)
  # Load trained model
  model = load_model(model_name)
  image_files = list(paths.list_images(read_path))
  predictions = np.ndarray((len(image_files),3), dtype=int)
  # Load binarizer so that we can invert one-hot encoding
  with open(one_hot, "rb") as f:
      lb = pickle.load(f)

  # Number of images to classify(files in 'read_path' folder)
  total = len(image_files)
  for img in image_files:
      imstr = str(img)
      # Get the number in the file name
      rel_path = re.search('_(\d*)', imstr)
      # Casting to int will remove non-significant integers e.g int(0001) -> 1
      idx = int(rel_path.group(1)) -1
      img = cv2.imread(img)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # Median filter to remove noise
      median = cv2.medianBlur(img, 5)
      # Convert to binary image using threshholding
      ret,thresh1 = cv2.threshold(median ,127,255,cv2.THRESH_BINARY)
      im = thresh1
      # Get the contours of solid objects in the image and how they relate to eachother
      contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      letter_regions = []
      area_bound = 200
  # Size of entire image
      th, tw = im.shape
  # Counter to index into hierarchy
      i = 0
      for contour in contours:
          x, y, w, h = cv2.boundingRect(contour)
          # Don't include the entire image, elements who are to small or countours which have a parent countour except for the contour of the image itself
          if (w*h > area_bound and w*h < th*tw and hierarchy[0][i][3] <= 0):
              if w / h > 1:
          # Asumed to be two letters stuck together
                  half = int(w / 2)
          # Left half
                  letter_regions.append((x, y, half, h))
          # Right half
                  letter_regions.append((x + half, y, half, h))
              else:
                  letter_regions.append((x,y,w,h))
          i += 1
      # Don't use incorrectly segmented data
      if(len(letter_regions) != 3):
          # If segmentation fails to find exactly three letters, just guess.
          predictions[idx,:] = np.random.randint(3, size=3)
          print(idx)
          print(len(letter_regions))
          continue
      pred = []
      # Letters appear in increasing order of their x-coordinate
      letter_regions = sorted(letter_regions, key=lambda x: x[0])
      for i in range(len(letter_regions)):
          x, y, w ,h = letter_regions[i]
          # Slice out the letter from the original image
          letter_image = im[y - 2:y + h + 2, x - 2:x + w + 2]
          # Resize image to fit our network
          image = cv2.resize(letter_image, (20, 20))
          # Add a channels or else keras complain
          image = np.expand_dims(image, axis=2)
          image = np.expand_dims(image, axis=0)
          prediction = model.predict(image)
          # Revert one-hot encoding
          prediction = lb.inverse_transform(prediction)[0]
          predictions[idx,i] =  int(prediction)

  return predictions
