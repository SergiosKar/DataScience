'''forecast bitcoin'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


series = pd.Series.from_csv('bitcoinallhistory.csv', header=0, parse_dates=[0], index_col=0)



'''RNN'''

#training batches
TS = np.array(series)

num_periods = 10
f_horizon = 1  #forecast horizon, one period into the future

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1, num_periods, 1)

y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1, num_periods, 1)

print (x_batches.shape)
print (y_batches.shape)

plt.plot(np.ravel(y_batches))

plt.show()

def test_data(series,forecast,num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, num_periods, 1)
    testY = TS[-(num_periods):].reshape(-1, num_periods, 1)
    return testX,testY

X_test, Y_test = test_data(TS,f_horizon,num_periods )
print (X_test.shape)
print (Y_test.shape)


#model
tf.reset_default_graph()


inputs = 1
hidden = 100
output = 1

X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)  # create our RNN object
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)  # choose dynamic over static

learning_rate = 0.001

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

loss = tf.reduce_sum(tf.square(outputs - y))  #
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(
    loss)

init = tf.global_variables_initializer()

epochs = 1000  # number of iterations or training cycles, includes both the FeedFoward and Backpropogation




with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(ep, "\tMSE:", mse)

    y_pred = sess.run(outputs, feed_dict={X: X_test})
    print(y_pred)

plt.title("Forecast vs Actual", fontsize=14)
plt.plot(np.ravel(Y_test))

plt.plot(np.ravel(y_pred))
plt.xlabel("Time Periods")

plt.show()