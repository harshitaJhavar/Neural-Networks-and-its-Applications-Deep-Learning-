import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from plotly.graph_objs import Scatter, Data
from plotly.offline import plot

from scipy.optimize import curve_fit


neurons_count = 2048


def weight_variable(shape):
    # Since we're using ReLU neurons, it is also good practice to initialize them
    # with a slightly positive initial bias to avoid "dead neurons".
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # just constant
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Import data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# 28x28 pixels, 1 color, -1 means calculate automatically this dimension such that
# in total we have 784 elements
x_image = tf.reshape(x, [-1, 28, 28, 1])


# Define architecture of CNN

# 1-st convolutional layer
# 5x5 patch, 1 color (input channel), 32 features (output channel)
W_conv1 = weight_variable([5, 5, 1, 32])
# 32 features requires 32 biases
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 28x28x32
h_pool1 = avg_pool_2x2(h_conv1)  # 14x14x32


# Fully-connected layer
# 7x7x64=3136 neurons -> 1024 neurons (fully-connected, usual matrix multiplication)
W_fc1 = weight_variable([14 * 14 * 32, neurons_count])
b_fc1 = bias_variable([neurons_count])

h_pool1_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 32])  # 1x3136
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)  # 1x1024


# Readout layer
# 1024 learned features and 10 output digits
W_fc2 = weight_variable([neurons_count, 10])
b_fc2 = bias_variable([10])

# TODO: experiment with W_fc1
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2  # 1x10


# Define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# Start session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
train_accuracy = []
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_accuracy.append(accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))
    print("step %d, training accuracy %g" % (i, train_accuracy[-1]))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


def fit_f(x, a, b, c, d):
    return a*x**0.5 + b*x**0.25 + c*x**0.125 + d*x**0.0625

fit_args, _ = curve_fit(fit_f, list(range(len(train_accuracy))), train_accuracy)
train_accuracy_smoothed = [fit_f(x, *fit_args) for x in range(len(train_accuracy))]
plot(Data([
    Scatter(y=train_accuracy, name='Train accuracy'),
    Scatter(y=train_accuracy_smoothed, line=dict(shape='spline'), name='Train accuracy (smoothed)'),
]), image='svg')


print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
