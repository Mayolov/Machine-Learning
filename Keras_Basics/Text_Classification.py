# Creates a model that does sentiment analysis on the IMDB dataset
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import losses

#This is preprocessing the text to prevent training-testing skewing
# it is important to preprocess the data identically at train and test time
# the html tags in the text wont be removed in the TextVectorization layer
#(which converts text to lowercase and strips punctuation by default, but doesn't strip HTML).
#So here is where we write our own custom standardiztion functionto remove the html
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation),'')

#This checkss the result of using this layer to preprocess some data
def vectorize_text(text,label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


"""url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", origin=url, untar=True,
                                  cache_dir=".",cache_subdir="")
"""
dataset_dir = "/home/mayolo/Machine_Learning/aclImdb"

os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, "train")
print(os.listdir(train_dir))

"""
#used to read the data, dont need anymore
sample_file = os.path.join(train_dir, "pos/1181_9.txt")
with open(sample_file) as f :
    print(f.read())
"""

#It think this is whats removing this sectionof the file
# As the IMDB dataset contains additional folders, 
# you will remove them before using this utility.
remove_dir = os.path.join(train_dir, "unsup")
#shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

#text_dataset_from_directory helps prepare the data for training
#split the raw training statset to 80:20
raw_train_ds =tf.keras.utils.text_dataset_from_directory(
    '/home/mayolo/Machine_Learning/aclImdb/train',
    batch_size=batch_size,
    validation_split= 0.2,
    subset='training',
    seed=seed)

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print('review: ', text_batch.numpy()[i])
        print('Label: ', label_batch.numpy()[i])
        
print('label 0 corresponds to', raw_train_ds.class_names[0])
print('Label 1 corresponds to', raw_train_ds.class_names[1])

# Creating validation set
#make sure to either sepcify a random seed or pass shuffle=False
# so there is no overlap between the validation set and training set
raw_val_ds =tf.keras.utils.text_dataset_from_directory(
    '/home/mayolo/Machine_Learning/aclImdb/train',
    batch_size=batch_size,
    validation_split=.2,
    subset='validation',
    seed=seed)

#testing dataset
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    '/home/mayolo/Machine_Learning/aclImdb/test',
    batch_size=batch_size)

max_features = 10000
sequence_length = 100

#Text_vectorization allows you to standardize, tokenize, and vectorize
#This is the TextVectorization Layer, the ouput is set to intto create a unique integer indice for each token
# the output sequence length is a constant which is 100 characters
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length,)

# adapt fits the state of the preprocessing layer to the dataset
# ONLY USE TRAINING DATA WHEN CALLING ADAPT
#(which converts text to lowercase and strips punctuation by default, but doesn't strip HTML).
train_text =  raw_train_ds.map(lambda x,y:x)
vectorize_layer.adapt(train_text)

#retrieve a batch of 32 reviews and labes from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0],label_batch[0]

print("review: ", first_review)
print("Label:", raw_train_ds.class_names[first_label])
print("Vectorized review: ", vectorize_text(first_review,first_label))

#You can lookup the token (string) that each integer corresponds to by calling .get_vocabulary() on the layer.
print("   1 ----> ", vectorize_layer.get_vocabulary()[1])
print("1287 ----> ", vectorize_layer.get_vocabulary()[1287])
print(" 313-----> ", vectorize_layer.get_vocabulary()[313])
print( "Vocabulary size:{}".format(len(vectorize_layer.get_vocabulary())))

#As a final preprocessing step apply TextVecorization Layer 
# #To the  train, validatoin and test dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


AUTOTUNE = tf.data.AUTOTUNE

# cache are important to make sure the I/O doesnt become blocking
# Cache keep the data in memory after its loaded off disk
# this ensures the dataset wont bottleneck while training the model
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# prefetch overlaps data preprocessing and model execution while training
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# create the model NN
embedding_dim = 16

model = tf.keras.Sequential([
    # This layer takes the integer-encoded reviews and looks up an embedding vector for each word index
    # The vectors are learned as the model trains
    # The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding)
    layers.Embedding(max_features+1, embedding_dim),
    layers.Dropout(.2),
    # this layer returns a fixed-length output vector for each ex. by averaging
    #over the sequence dimension. This allows the model to handle input of var length
    #in the simplest way possible 
    layers.GlobalAveragePooling1D(),
    layers.Dropout(.2),
    # this is a densely connected layer with single output node
    layers.Dense(1)])

model.summary()

# Since this is a binary classification problem and the model outputs a probability 
# (a single-unit layer with a sigmoid activation), you'll use losses.BinaryCrossentropy loss function.
model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 11
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

loss, accuracy =  model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

# model.fit() returns a History object that
# contains a dictionary with everything that happened during training:
history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible...",
  "Bad is what this movie is not.", # This one failed but still is decent 
  "I wasted my time here.",
  "love",
  "love movie"
]

print(export_model.predict(examples))