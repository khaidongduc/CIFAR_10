import tensorflow as tf
import pickle

import model

x_train, y_train = pickle.load(open("data/train_data", "rb"))
x_test, y_test = pickle.load(open("data/test_data", "rb"))

model = model.create_model()

# compiling and training
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=25,
                    validation_data=(x_test, y_test))

# save model
model.save_weights("model/model.h5")
