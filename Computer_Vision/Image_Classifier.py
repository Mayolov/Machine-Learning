import matplotlib.pyplot as plt
import PIL
import pathlib
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential 

# dataset of 3k pictures of flowers
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count  = len(list(data_dir.glob('*/*.jpg')))
#print(image_count) #3670 pictures

# This pulls the data into a list 
#roses = list(data_dir.glob('roses/*'))
#PIL.Image.open(str(roses[0]))
#PIL.Image.open(str(roses[1]))

batch_size = 16
img_height = 180
img_width = 180

# split the validation data 80% training, 20% validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

#Visualize the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
#plt.show()

# manually iterates through the dataset and retrieves batches of images
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# image_batch is a tensor of the shape (32, 180, 180, 3)
# This is a batch of 32 images of shapre 180x180x3( the last dimension regers to color channels RGB)
# The label_batch is a tensor of the shape (32,), these are correspondinglabels to the 32 images.

# you can call .numpy() on the image_batch and labels_batch tensors to convert them to a numpy.ndarray

# Dataset.cache keeps the imgs in memory after they're loaded off disk during the first epoch. Ensures the
# dataset does not become a bottleneck while traiing the model

# Data.prefetch overlaps data preprocessing and mdoel execution while training

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# RGB channel values are in the [0,255] range, this is not ideal for a NN. 
# standardize values to be in the [0,1] range by using:
normalization_layer = layers.Rescaling(1./255)

# Applt it to the dataset by calling Dataset.map:
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x),y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image),np.max(first_image))

num_classes = len(class_names)

# This model consists of 3 convolution blocks (Conv2D) with a max pooling layer in eac of them
# Thers a fully connected layer (Dense) with 128 unites on top of it that is activated by a ReLU activation function
# not tuned for accuracy

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Dense(128, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(8, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()