import csv
import matplotlib.image as mpimg
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []

with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
      lines.append(line)

train_lines, validation_lines = train_test_split(lines, test_size=0.2)

def generator(lines, batch_size=32):
  num_lines = len(lines)
  while 1:
    shuffle(lines)
    for offset in range(0, num_lines, batch_size):
      batch_lines = lines[offset:offset+batch_size]
      images = []
      angles = []
      for batch_sample in batch_lines:
        name = './data/IMG/'+batch_sample[0].split('/')[-1]
        center_image = mpimg.imread(name)
        center_angle = float(batch_sample[3])
        flipped_center_image = np.fliplr(center_image)
        flipped_center_angle = -center_angle
        images.append(center_image)
        angles.append(center_angle)
        images.append(flipped_center_image)
        angles.append(flipped_center_angle)

      X_train = np.array(images)
      y_train = np.array(angles)
      yield shuffle(X_train, y_train)

train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_lines), validation_data=validation_generator, nb_val_samples=len(validation_lines), nb_epoch=10)
model.save('model.h5')
