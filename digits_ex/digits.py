import tensorflow as tf
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist #28x28 resolution - hand written digits from 0-9


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#input layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

#hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#output layer number is 10 because we have 10 classifications
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


#model training parameters optimize / loss and track accuracy

model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

#train model

model.fit(x_train, y_train, epochs =3)





#show array position 1
plt.imshow(x_train[0], cmap= plt.cm.binary)
plt.show()

