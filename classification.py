#!/usr/bin/env python

# Imports:
import cv2
import numpy as np
import random
import datetime as dat
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Function which performs facial expression classification
# Input:
#   image ->Croped image of detected face
#   expressions -> string matrix with expression labels
#   task_exp -> expression to be reached
# Output:
#   expression -> result of the classification (best match)
#              -> probability of best match classification
#   task_prob -> probability of task expression
def classify_expression(image, expressions, task_exp):

    # Resize croped face on 48x48 pixels
    detected_face =  cv2.resize(image, (48, 48)) #64
    # Convert it to float type and divide with 255 to perform the preprocessing
    detected_face = detected_face.astype('float') / 255.0
    # Convert image matrix to array (each array value to one oeuron from imput layer)
    img_pixels = img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    # Perform prediction
    predictions = model.predict(img_pixels)
    # Take index of maximal label
    max_index = np.argmax(predictions[0])

    # Take label
    expression = expressions[max_index]
    # Take task label probability
    task_prob = predictions[0][task_exp]

    return (expression, max(predictions[0]), task_prob)


# If not exist, create directory to save result images
if (os.path.isdir("results") == False):
    os.mkdir("results")

# Import frontal face haar cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load pretrained CNN model for expression classification
model = load_model("epoch_90.hdf5", compile=False)
#model = model_from_json(open("tensorflow-101/model/facial_expression_model_structure.json", "r").read())
#model.load_weights('tensorflow-101/model/facial_expression_model_weights.h5')

# Init video capture from WebCam
cap = cv2.VideoCapture(0)
#expressions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
# Expression labels
expressions = ('angry', 'scared', 'happy', 'sad', 'surprised', 'neutral')

# Open named window and set it to full screen
cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Init loop variables
now_min = 0 # Current minute
now_sec = 0 # Current second
now = 0 # Current time
random.seed(1) # Seed pseudo random generator
task_expression = random.randint(0, 5) # Take random expression from the list
save_counter = 0 # Counter to name saved images...

# Infinite loop:
while(True):
    
    # Capture the frame and convert it to grayscale
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 

    # On every 15th minute of an new hour, change task expression
    now = dat.datetime.now()
    now_min = now.minute
    now_sec = now.second
    if now == 15 & now_sec == 1:
        task_expression = random.randint(0,5)

    # Put task text in the corner
    cv2.putText(frame, "Try to be: " + expressions[task_expression], 
            (int(10), int(50)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2)

    # Detect faces using multiscale
    faces = face_cascade.detectMultiScale(gray,
            scaleFactor=1.3, 
            minNeighbors=10, 
            minSize = (int(0.1*cap.get(3)), int(0.1*cap.get(4))))

    # For every detected face
    for(x, y, w, h) in faces:
        # Put rectange around it
        cv2.rectangle(frame, 
                (x, y), 
                (x+w, y+h), 
                (255,0,0), 2)
    
    # If 'a' is pressed
    if cv2.waitKey(1) & 0xFF == ord('a'):
        # Loop for every detected face
        for (x, y, w, h) in faces:
            # Put rectangle
            cv2.rectangle(frame, 
                    (x, y), 
                    (x+w, y+h), 
                    (255,0,0), 2)
            # Execute function to classify expression of detected face
            emotion, prob, task_prob = classify_expression(gray[int(y):int(y+h), int(x):int(x+w)], # Grayscale image croped on detected face rectangle
                expressions,                                                                       # Expression labels
                task_expression)                                                                   # Task expression
            # Put text for primary detected expression with probability on top of the rectangle
            cv2.putText(frame, emotion + " " + str(int(prob*100)) + "%", 
                    (int(x), int(y)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2)
            # Put task expression label and its probability below the rectangle
            cv2.putText(frame, expressions[task_expression] + " " + str(int(task_prob*100)) + "%",
                    (int(x), int(y)+h+30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
       
        # Show frame with all rectangles and labels
        cv2.imshow('frame', frame)
        # Wait for frame to be displayed and make delay to enable user to see results (2 sec in sum)
        cv2.waitKey(2000)
        # Save image in result direcotry
        cv2.imwrite("results/slika" + str(save_counter) + ".jpg", frame)
        save_counter += 1
    
    # Show frame only with rectangles
    cv2.imshow('frame', frame)

    # If 'q' is pressed get out of the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()
# Destroy webcam window
cv2.destroyAllWindows()

