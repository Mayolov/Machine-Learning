import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import losses

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation),'')

def vectorize_text(text,label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label




url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", origin=url, untar=True,
                                  cache_dir=".",cache_subdir="")

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
remove_dir = os.path.join(train_dir, "pos")
shutil.rmtree(remove_dir)
batch_size = 32
seed = 42

raw_train_ds =tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=.2,
    subset='training',
    seed=seed)

for text_batch, label_batch in raw_train_ds:
    for i in range(3):
        print('review: ', text_batch.numpy()[i])
        print('Label: ', label_batch.numpy()[i])
        
print('label 0 corresponds to', raw_train_ds.class_names[0])
print('Label 1 corresponds to', raw_train_ds.class_names[1])

raw_val_ds =tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=.2,
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size)

max_features = 10000
sequence_length = 100

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length,)

train_text =  raw_train_ds.map(lambda x,y:x)
vectorize_layer.adapt(train_text)

text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0],label_batch[0]

print("review: ", first_review)
print("Label:", raw_train_ds.class_names[first_label])
print("Vectorized review: ", vectorize_text(first_review,first_label))

print("1287 ----> ", vectorize_layer.get_vocabulary()[1287])
print(" 313-----> ", vectorize_layer.get_vocabulary()[313])
print( "Vocabulary size:{}".format(len(vectorize_layer.get_vocabulary())))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

model = tf.keras.Sequential([
    layers.Embedding(max_features+1, embedding_dim),
    layers.Dropout(.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(.2),
    layers.Dense(1)])

model.summary()

model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 10
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

loss, accuracy =  model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

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
  "The movie was terrible..."
]

export_model.predict(examples)