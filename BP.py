import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

from tensorflow.python.eager.context import PhysicalDevice
from tensorflow.python.ops.gen_array_ops import pad
warnings.simplefilter(action='ignore', category=FutureWarning)


# pokud máme GPU tak jej lze aktivovat následujícím kódem

# Physical_Devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(Physical_Devices))
# tf.config.experimental.set_memory_growth(Physical_Devices[0], True)

# kategorizace obrázků


# zpracování dat do požadovné struktury

# rozdělení do skupin Pes/Kočka a do sad kde jsou sady trénovací, validační a testovací



# zpracování dat 

train_path = 'C:/Users/janpe/Desktop/school/SEM5/BP/kod/third/Datasets/Datasets/Train'
valid_path = 'C:/Users/janpe/Desktop/school/SEM5/BP/kod/third/Datasets/Datasets/Valid'
test_path = 'C:/Users/janpe/Desktop/school/SEM5/BP/kod/third/Datasets/Datasets/Test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=train_path, target_size=(224,224), classes=['car', 'truck'], batch_size=10)

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['car', 'truck'], batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=test_path, target_size=(224,224), classes=['car', 'truck'], batch_size=10, shuffle=False)



assert train_batches.n == 726 # ověřujeme že je trénovacích vzorků určitý počet
assert valid_batches.n == 42 # ověřujeme že je ověřovacích vzorků určitý počet
assert test_batches.n == 20 # ověřujeme že je testovacéch vzorků určitý počet
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2 # ověřujeme že počet tříd je všude stejný a to 2


imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1,10)
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# pro náhled lze odkomentovat následující instrukce
# nahlížíte na sadu která reprezentuje jeden vzorek
# 
#  
# plotImages(imgs) # POZOR! program nebude pokračovat dokud nebude okno z kontrolními zvířátky zavřen
# print(labels)


model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224,224,3)),  #224 je velikost x 224 je velikost y a 3 je počet barevných kanálů (RGB = 3)
    MaxPool2D(pool_size=(2, 2), strides=2 ),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2 ),

    Flatten(),
    Dense(units=2, activation='softmax')

])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=8, verbose=2)


# Predict

test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)
print(test_labels)

test_batches.classes

predictions = model.predict(x=test_batches, verbose=0)

np.round(predictions)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))




def plot_confudion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This Function prints and plots the confusion matrix
    Normalization can by applied by seting 'normalize = True'.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("normalized confusion matrix")
    else:
        print("confusion matrix, without normalized")

    print(cm)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

test_batches.class_indices

cm_plot_labels = ['cat','dog']
plot_confudion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

