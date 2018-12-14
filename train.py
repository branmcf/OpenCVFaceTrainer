import os
import sys
import cv2
import json
import base64
import requests
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp

rootdir = '/Users/brandonmcfaraland/Desktop/et_train/lfw'
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


with open('processed_images.json') as infile:
    processed_image_obj = json.load(infile)

processed_image_keys = processed_image_obj.keys()
face_crop_obj = {}
skin_tones = ['','light', 'medium light', 'medium', 'medium dark', 'dark']
count = 0

# loop over the images in the test folder
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file == '.DS_Store':
            continue
        elif file in processed_image_keys:
            continue
        elif count < 3:
            path = os.path.join(subdir, file)

            # load the image
            image = cv2.imread(path)

            # convert image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_copy = image

            # convert image to gray
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # detect faces
            faces = haar_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

            for f in faces:
                x, y, w, h = [ v for v in f ]
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                face_crop_obj[file] = image[y:y+h, x:x+w]
            count +=1

face_crop_keys = face_crop_obj.keys()
for filepath in face_crop_keys:
    face = face_crop_obj[filepath]
    encoded_string = ''
    plt.imshow(face)
    plt.show(block=False)
    given_label = raw_input("What skin tone is this?")
    if given_label == "close":
        with open('processed_images.json', 'w') as outfile:
            json.dump(processed_image_obj, outfile)
        exit()
    else:
        face_c = face.copy(order='C')
        face_c = cv2.cvtColor(face_c, cv2.COLOR_BGR2RGB)
        cv2.imwrite("current_face.png", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        with open("current_face.png", "rb") as image_file:
            encoded_face_string = base64.b64encode(image_file.read())
        current_face_array = [encoded_face_string]
        given_label = int(given_label)
        data = {
                "dsid": 1,
                "feature": current_face_array,
                "label": skin_tones[given_label],
                "modelName":0
        }
        r = requests.post("http://10.8.126.46:8000/AddDataPoint", data=json.dumps(data))
        r2 = requests.get("http://10.8.126.46:8000/UpdateModel?dsid=1&modelName=0")
        processed_image_obj[filepath] = 'processed'
    plt.close()

with open('processed_images.json', 'w') as outfile:
    json.dump(processed_image_obj, outfile)