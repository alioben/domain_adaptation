## This code is only used for pre-training the source stream on source data

import tensorflow as tf
from helpers import *
import os
import cv2

# Model parameters
xydim, cdim = 28, 1
batch_size = 2048
n_classes = 10
checkpoint_dir = 'checkpoints'

# Placeholders for the input and labels
X_source = tf.placeholder(tf.float32, shape=[batch_size, xydim, xydim, cdim])
Y_source = tf.placeholder(tf.float32, shape=[batch_size, n_classes])

''' Weights for the source stream '''
wS = {
    "conv1": tf.get_variable('conv1', shape=[5, 5, cdim, 20], initializer=tf.contrib.layers.xavier_initializer()),
    "conv2": tf.get_variable('conv2', shape=[5, 5, 20, 50], initializer=tf.contrib.layers.xavier_initializer()),
    
    "fc1": tf.get_variable('fc1', shape=[int(50*(xydim/4)*(xydim/4)), 500], initializer=tf.contrib.layers.xavier_initializer()),
    "classifier": tf.get_variable('classifier', shape=[500, n_classes], initializer=tf.contrib.layers.xavier_initializer())
}

def stream(X, w):
    ''' Creates a small ~LeNet network '''
    model = conv2d_relu(X, w['conv1'])
    model = max_pool2d(model)
    model = conv2d_relu(model, w['conv2'])
    model = max_pool2d(model)

    model = tf.reshape(model, [batch_size, int(50*(xydim/4)*(xydim/4))])
    model = dense_relu(model, w['fc1'])
    return model

def classifier_logits(features):
	''' Performs the classification on a feature vector '''
	return  tf.matmul(features, wS['classifier'])

# Get the source and target features
source_features = stream(X_source, wS)
source_logits = classifier_logits(source_features)
class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_source, logits=source_logits))

# Accuracy 
prob = tf.nn.softmax(source_logits)
prediction = tf.argmax(prob, 1)
correct_answer = tf.argmax(Y_source, 1)
equality = tf.equal(prediction, correct_answer)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

# Initilize parameters and optimizer
optimizer = tf.train.AdamOptimizer().minimize(class_loss)
session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver(var_list=list(wS.values()))
if os.path.exists(checkpoint_dir+'/pretrained_svhn.ckpt.meta'):
	saver.restore(session, checkpoint_dir+"/pretrained_svhn.ckpt")
	print('[Info] Loaded latest model.')

# Training loop
max_iters = 1000000
for it in range(max_iters):
	# Load a minibatch
	x_batch_source, y_batch_source, x_t, _ = minibatch_svnh_mnist(batch_size)
    
	# Do one step of optimization
	_, class_loss_curr = session.run([optimizer, class_loss], feed_dict={X_source:x_batch_source, Y_source:y_batch_source})

	if it%1 == 0:
		print("[Info] Iteration {}, class_loss: {:.4}".format(it, class_loss_curr))

	if it%50 == 0:
		# Compute the training/validation accuracies
		x_batch_source_test, y_batch_source_test, _, _ = minibatch_svnh_mnist(batch_size, type_='test')
		training_accuracy = session.run(accuracy, feed_dict={X_source:x_batch_source, Y_source:y_batch_source})
		testing_accuracy = session.run(accuracy, feed_dict={X_source:x_batch_source_test, Y_source:y_batch_source_test})
		print("[Info] Iteration {}, train_acc: {:.3}, val_acc: {:.3}".format(it, training_accuracy, testing_accuracy))

		# Save the current model
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		saver.save(session, checkpoint_dir+"/pretrained_svhn.ckpt")
        
		if abs(testing_accuracy-training_accuracy) > 0.04 and testing_accuracy>0.9:
			exit()
