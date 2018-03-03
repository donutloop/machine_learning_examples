
'''
input > weiht > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost function (cross entropy) > optimization function (optimizer) > minimze cost (AdamOptimizer, SGG, AdaGrad) 

backpropagation 

feed forward + backprop = epoch
'''

import os
import tensorflow as tf 
import gzip
import os
import tempfile
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mnist = read_data_sets("/tmp/data/", one_hot=True)

'''
10 classes, 0-9

one hot 
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
...
'''

# hidden layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# 10 classes, 0-9
n_classes = 10

# bacth size of featuresets
batch_size = 100

# flatten out to 784 pixels by second parameter
x = tf.placeholder('float32')
y = tf.placeholder('float32')

def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases for each layer
    
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output
    

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    # default learning rate for optimizer = 0.01
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 

        for epoch in range(hm_epochs):
            epoch_loss = 0 
            for _ in range(int(mnist.train.num_examples/batch_size)):           
                epoch_x , epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c 

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss) 

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y, 1))     

        arrcuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', arrcuracy.eval({x:mnist.test.images, y: mnist.test.labels}))              


train_neural_network(x)





