import numpy as np
import tensorflow as tf

x_train = np.random.random((50, 730, 1))
ts_train = np.random.random((50, 730, 3))
y_train = np.random.random((50, 5))

ds = tf.data.Dataset.from_tensor_slices(((x_train, ts_train), y_train))

for (x, t), y in ds.take(1):
  print(x.shape, t.shape, y.shape)
(730, 1) (730, 3) (5,)
And here is an example model:

input1 = tf.keras.layers.Input((730, 1))
input2 = tf.keras.layers.Input((730, 3))
x = tf.keras.layers.Flatten()(input1)
y = tf.keras.layers.Flatten()(input2)
outputs = tf.keras.layers.Concatenate()([x, y])
outputs = tf.keras.layers.Dense(5)(outputs)
model = tf.keras.Model([input1, input2], outputs)
model.compile(optimizer='adam', loss='mse')
model.fit(ds.batch(10), epochs=5)





from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization,Flatten
from keras.optimizers import adam
import tensorflow as tf
import numpy as np

a = tf.constant([[1,2,3,4,4,5,61,2,3,4,4,5,6],[1,2,3,4,4,5,61,2,3,4,4,5,6]], dtype=tf.float32)
y = tf.constant([[1,1,1,0,0,0,0,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0,0]], dtype=tf.float32)
model = Sequential([
   
    Dense(20, activation="relu"),
    Dense(100, activation="relu"),
    Dense(1, activation="sigmoid")
])

print(np.shape(a))

def generator(a,y):
    while True:
        for i in np.random.permutation(len(a)):
            yield a[i], y[i]


            
dataset = tf.data.Dataset.from_generator(generator, args=(a,y),     
                                         output_signature=(  
                                             tf.TensorSpec(shape=( 13,), 
                                             dtype=tf.float32), 
                                             tf.TensorSpec(shape=(1,), dtype=tf.float16))
                                        )

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    my_opt = adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=my_opt, metrics=['accuracy'])

    model.fit(dataset, epochs=3)
