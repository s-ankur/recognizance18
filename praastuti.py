import numpy as np
import pandas as pd
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

image_dim = 28 * 28
batch_size = 50
epochs = 10


def load_data(path):
    df = pd.read_csv(path)
    df = df.values
    images = []
    labels = []
    for i in range(df.shape[0]):
        images.append(df[i][1:])
        labels.append(df[i][0])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def modeldense():
    input_img = Input(shape=(image_dim,))
    encoded = Dense(512, activation='relu', )(input_img)
    encoded = Dense(40, activation='relu', )(encoded)
    encoded = Dense(10, activation='relu', )(encoded)
    encoded = Model(input_img, encoded)
    return encoded


model1 = modeldense()

X_train, Y = load_data(r'train.csv')
Y_train = np.zeros((Y.shape[0], 10))
Y_train[np.arange(Y_train.shape[0]), Y] = 1
X_train = X_train / 255
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.6, random_state=42)

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model1.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(X_test, Y_test))

model1.evaluate(X_test, Y_test)


# Loss Curves
def f():
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
