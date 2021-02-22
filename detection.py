import numpy as np 
import cv2
from scipy.spatial import distance
import os
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MIN_DISTANCE = 140 # Estimation for 2 Meters
input_shape = (128,128,3);
num_epoch = 10
batch_size = 20
model_name = 'mask_model.h5'
load_model = 1 # yes -> '1'
red_color = (0,25,255)
green_color = (0,255,0)

# Loading dataset
train_set = 'data/face-mask-dataset/Train'
test_set = 'data/face-mask-dataset/Test'
validation_set = 'data/face-mask-dataset/Validation'

# Face Detetction Model
face_model = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
body_model = cv2.CascadeClassifier('data/haarcascades/haarcascade_fullbody.xml')

# Mask Detection Model 
if load_model == 1:
    model = keras.models.load_model(model_name)
else:
    # Data augmentation
    train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2)
    train_generator = train_datagen.flow_from_directory(directory=train_set, target_size=(128,128), class_mode='categorical', batch_size=batch_size)

    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_datagen.flow_from_directory(directory=test_set, target_size=(128,128), class_mode='categorical', batch_size=batch_size)

    val_datagen = ImageDataGenerator(rescale=1.0/255)
    val_generator = val_datagen.flow_from_directory(directory=validation_set, target_size=(128,128), class_mode='categorical', batch_size=batch_size)

    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(7, 7)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="sigmoid"))

    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")

    model.fit_generator(generator=train_generator, steps_per_epoch=len(train_generator)//batch_size, epochs=num_epoch, 
                        validation_data=val_generator, validation_steps=len(val_generator)//batch_size)
    # Save Model
    model.save(model_name)

    # Evaluate Model
    model_evaluate = model.evaluate_generator(test_generator)
    print(model_evaluate)

# Setting up Camera
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

# Detection Labels
mask_label = {0:'NO MASK', 1:'MASK'}
mask_label_color = {0:red_color, 1:green_color}
dist_rect_color = {0:green_color, 1:red_color}

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale Input
    faces = face_model.detectMultiScale(
        gray,     
        scaleFactor=1.2, # Reduce Image Size At Each Scale
        minNeighbors=5,     
        minSize=(20, 20) # Minimum Rectangle Size
    ) 

    label = [0 for i in range(len(faces))]
    for i in range(len(faces)-1):
        # Social Distance 
        for j in range(i+1, len(faces)):
            # Calculate distance between points
            dist = distance.euclidean(faces[i][:2], faces[j][:2])
            print("Distance: {} between {} and {}".format(dist, faces[i][:2], faces[j][:2]))

            # Mark distance violation with '1'
            if dist<MIN_DISTANCE:
                label[i] = 1
                label[j] = 1

    # Detected Faces
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        mask_pred = img[y:y+h,x:x+w]
        mask_pred = cv2.resize(mask_pred,(128,128))
        mask_pred = np.reshape(mask_pred,[1,128,128,3])/255.0
        check_mask = model.predict(mask_pred)
        cv2.putText(img,mask_label[check_mask.argmax()], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, mask_label_color[check_mask.argmax()], 4)
        cv2.rectangle(img,(x,y), (x+w,y+h), dist_rect_color[label[i]], 1)

    cv2.imshow('Feed',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()