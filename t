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




