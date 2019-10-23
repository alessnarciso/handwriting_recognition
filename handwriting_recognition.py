#Handwriting Recognition
#Exercise 2 from Coursera Intro to Tensorflow
#Specifications:
#1. It should succeed in less than 10 epochs
#2. When it reaches 99% or greater it should print out the string 
  #"Reached 99% accuracy so cancelling training!"

import tensorflow as tf
mnist = tf.keras.datasets.mnist

#target variable
target_accuracy = 99

#neuron number
num_neurons = 512

#call back function
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<(1.0-float(target_accuracy/100))):
      print("\nReached ", target_accuracy,"% accuracy so cancelling training!")
      self.model.stop_training = True

#load MNIST library
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#normalize pixels 
x_train = x_train / 255.0
x_test = x_test / 255.0

callbacks = myCallback()

#1st layer: flatten matrix to linear array
#2nd layer: neurons
#3rd layer: 10 numbers to classify - 0 to 9
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), 
    tf.keras.layers.Dense(num_neurons, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


print("\n***Compiling Model***: \n")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model fitting / training
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])


