import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from convert_data_to_images import convert_to_image

x_train, Y = convert_to_image(r"train.csv")
Y_train = np.zeros((Y.shape[0], 10))
Y_train[np.arange(Y_train.shape[0]), Y] = 1
print(Y_train)
X_train = []
for i in range(x_train.shape[0]):
    X_train.append(x_train[i].reshape(28, 28, 1))
X_train = np.array(X_train)
X_train = X_train / 255


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    ##    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    ##    model.add(Conv2D(64, (3, 3), activation='relu'))
    ##    model.add(MaxPooling2D(pool_size=(2, 2)))
    ##    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.6, random_state=42)
model1 = create_model()
batch_size = 50
epochs = 10
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model1.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(X_test, Y_test))

model1.evaluate(X_test, Y_test)
# Loss Curves
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
