import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

image_path = './data/IMG/'
angle_adjustment = 0.1

total_left_angles = 0
total_right_angles = 0
total_straight_angles = 0

images = []
angles = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    k = 0
    for line in reader:
        k += 1
        if k==1: continue
        center_image = cv2.imread(image_path + line[0].split('/')[-1])
        center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

        images.append(center_image_rgb)
        angles.append(float(line[3]))

        #flipped
        images.append(cv2.flip(center_image_rgb, 1))
        angles.append(-float(line[3]))

        left_image = cv2.imread(image_path + line[1].split('/')[-1])
        left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        images.append(left_image_rgb)
        angles.append(float(line[3])+angle_adjustment)
        #flipped
        images.append(cv2.flip(left_image_rgb, 1))
        angles.append(-(float(line[3])+angle_adjustment))

        right_image = cv2.imread(image_path + line[2].split('/')[-1])
        right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        images.append(right_image_rgb)
        angles.append(float(line[3])-angle_adjustment)
        #flipped
        images.append(cv2.flip(right_image_rgb, 1))
        angles.append(-(float(line[3])-angle_adjustment))

        if(float(line[3]) < -0.15):
            total_left_angles += 1
        elif(float(line[3]) > 0.15):
            total_right_angles += 1
        else:
            total_straight_angles += 1


left_to_straight_ratio = total_straight_angles/total_left_angles
right_to_straight_ratio = total_straight_angles/total_right_angles

print('Total Samples : ', len(images))
print()
print('Initial Angle Distribution')
print('Total Left Angles : ', total_left_angles)
print('Total Right Angles : ', total_right_angles)
print('Total Straight Angles : ', total_straight_angles)
print('Left to Straight Ratio : ', left_to_straight_ratio)
print('Right to Straight Ratio : ', right_to_straight_ratio)

X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.1)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

train_samples_size = len(X_train)
validation_samples_size = len(X_val)

total_left_angles = 0
total_right_angles = 0
total_straight_angles = 0

for train_sample in y_train:
    if(float(train_sample) < -0.15):
        total_left_angles += 1
    elif(float(train_sample) > 0.15):
        total_right_angles += 1
    else:
        total_straight_angles += 1

left_to_straight_ratio = 0
right_to_straight_ratio = 0

left_to_straight_ratio = total_straight_angles/total_left_angles
right_to_straight_ratio = total_straight_angles/total_right_angles

print()
print('Train Sample Size : ', train_samples_size)
print('Validation Sample Size : ', validation_samples_size)

print()
print('After TTS, Angle Distribution')
print('Total Left Angles : ', total_left_angles)
print('Total Right Angles : ', total_right_angles)
print('Total Straight Angles : ', total_straight_angles)
print('Left to Straight Ratio : ', left_to_straight_ratio)
print('Right to Straight Ratio : ', right_to_straight_ratio)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data = (X_val, y_val), nb_epoch=10, shuffle=True, validation_split=0.1)

model.save('model.h5')
