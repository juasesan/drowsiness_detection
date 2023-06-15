import itertools
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report

from utils import plot_confusion_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'

data_dir = './Data/train_eyes'
batch_size = 16
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    validation_split=0.2,
    subset="training",
    seed=100,
    image_size=(img_height, img_width),
    batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    validation_split=0.2,
    subset="validation",
    seed=100,
    image_size=(img_height, img_width),
    batch_size=batch_size)


class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Model definition

num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(units=100, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# Model training
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Saving history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


sns.set_theme(palette="ch:s=.25,rot=-.25")
fig,ax = plt.subplots(figsize=(8,8))
sns.lineplot(data=loss, ax = ax, color="b", label='Training Loss')
sns.lineplot(data=val_loss, ax = ax, color="r", label='Validation Loss')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.savefig("./Figures/loss_result.png")

sns.set_theme(palette="ch:s=.25,rot=-.25")
fig,ax = plt.subplots(figsize=(8,8))
sns.lineplot(data=acc, ax = ax, color="b", label='Training accuracy')
sns.lineplot(data=val_acc, ax = ax, color="r", label='Validation accuracy')
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
plt.savefig("./Figures/accuracy_result.png")


model.save('./Models/CNN_model.h5')

tf.keras.utils.plot_model(model, to_file="./Figures/model_architecture.png", show_shapes=True, show_layer_names=True)

# Model testing
test_dir = './Data/test_eyes'
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    seed=100,
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size)

Y_test = np.array([])
for image_batch, labels_batch in test_ds:
  labels = np.array(labels_batch)
  Y_test = np.append(Y_test, labels)

Y_test = Y_test.astype(int)


predictions = model.predict(test_ds)
Y_pred = np.argmax(predictions,axis=-1)


cm = confusion_matrix(y_true= Y_test, y_pred=Y_pred)
cm_plot_labels=["Closed", "Open"]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")

print(classification_report(Y_test,y_pred=Y_pred))


