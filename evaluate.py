import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pickle
import numpy as np

import model

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


x_test, y_test = pickle.load(open("data/test_data", "rb"))
model = model.create_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
model.load_weights("model/model.h5")
model.evaluate(x_test, y_test)


predictions = np.argmax(model.predict(x_test), axis=1)

# plot the first 25 predictions
plt.figure(figsize=(10, 10))
for i in range(25):
    prediction = predictions[i]
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel("E {}, A {}".format(
        class_names[y_test[i][0]], class_names[prediction]))
    plt.imshow(x_test[i], cmap=plt.cm.binary)
plt.show()
