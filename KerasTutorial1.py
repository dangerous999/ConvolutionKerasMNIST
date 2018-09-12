import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

tf.keras.backend.set_image_data_format('channels_last')

mnist = tf.keras.datasets.mnist # 28 x 28 images digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

input_shape = (28, 28, 1)

x_train = x_train.reshape(60000,28,28,1)
x_test_old = x_test
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, 10) # One hot
y_test_old = y_test
y_test =  tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.Sequential()
model.add(  tf.keras.layers.Convolution2D( filters = 16, 
                                    kernel_size = (5, 5),
                                    strides = 1,
                                    padding = 'valid',
                                    activation = 'relu',
                                    input_shape = (28, 28, 1),
                                    name = 'conv1'
                                     ))
model.add ( tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, name = 'MaxPool1') )

model.add(  tf.keras.layers.Convolution2D( filters = 32, 
                                    kernel_size = 5,
                                    strides = 1,
                                    padding = 'valid',
                                    activation = 'relu', name = 'conv2' ))
model.add ( tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, name = 'MaxPool2') )

model.add(tf.keras.layers.Dense(256, activation = 'relu', name = 'Dense1'))
model.add(tf.keras.layers.Dense(256, activation = 'relu', name = 'Dense2'))
model.add(tf.keras.layers.Dense(128, activation = 'relu', name = 'Dense3'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense( 10, activation = 'softmax', name = 'output'))

model.compile( optimizer = 'adam', 
               loss = tf.keras.losses.categorical_crossentropy,
               metrics = ['accuracy']
              )
model.fit(x_train, y_train, epochs = 3, batch_size = 128)

new_model = tf.keras.models.load_model('epic_num_reader.model')
new_model.summary()

val_loss, val_acc = model.evaluate(x_test, y_test)
print ("Validation loss: ", val_loss, "Validation accuracy: ", val_acc)

predictions = model.predict([x_test])
for i in range (0, 10000):                    # show wrong predictions
  if np.argmax(predictions[i]) != y_test_old[i] : 
    print ( "Prediction: ", np.argmax( predictions[i]), " True value: ", y_test_old[i] )
    plt.imshow(x_test_old[i])
    plt.show()