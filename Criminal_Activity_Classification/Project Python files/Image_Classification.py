# import the necessary packages
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Notes 
# The Windows command prompt (cmd.exe) allows the ^ (Shift + 6) character to be used
# to indicate line continuation. It can be used both from the normal command prompt (which
# will actually prompt the user for more input if used) and within a batch file.


# Run model on cmd/anaconda prompt
# By -m specify the model directory
# By -l specify the label directory
# By -i specify the input image directory 
# E.g given below
# Terminal commands: python Classification.py -m CriminalActivityN.model -l lb.Npickle -i arson1.jpg
# Construct the argument parser and parse the arguments 

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output serialized model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-i", "--input", required=True,
                help="path to input image")
args = vars(ap.parse_args())


# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")

model = load_model(args["model"])

lb = pickle.loads(open(args["label_bin"], "rb").read())
print("[INFO] loading complete")


# load the input image and then clone it so we can draw on it later
image = cv2.imread(args["input"])
output = image.copy()
output = imutils.resize(output, width=400)
# our model was trained on RGB ordered images but OpenCV represents
# images in BGR order, so swap the channels, and then resize to
# 224x224 (the input dimensions for VGG16)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
# convert the image to a floating point data type and perform mean
# subtraction
image = image.astype("float32")
mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
image -= mean

preds = model.predict(np.expand_dims(image, axis=0))[0]
i = np.argmax(preds)
label = lb.classes_[i]
# draw the prediction on the output image
text = "{}: {:.2f}%".format(label, preds[i] * 100)
cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)