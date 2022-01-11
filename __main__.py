import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def train():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    evaluate(model)
    model.save('mnist.model')
    print("training complete")


def evaluate(model):
    accuracy, loss = model.evaluate(x_test, y_test)
    print("accuracy: ", accuracy)
    print("loss: ", loss)


def predict():
    model = tf.keras.models.load_model('mnist.model')
    predictions = model.predict([x_test])
    print(np.argmax(predictions[0]))
    plt.imshow(x_test[0], cmap=plt.cm.binary)
    plt.show()


# train()
# evaluate(tf.keras.models.load_model('mnist.model'))
