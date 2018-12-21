import tensorflow as tf
from helpers import *
from models.lenet import LeNet
from models.residualnet import ResidualNet
import os
import numpy as np

# Model parameters
xydim, cdim = 28, 1
batch_size = 128
n_classes = 10
feature_dim = 500
checkpoint_dir='./checkpoints'

# Placeholders for the input and labels
X_source = tf.placeholder(tf.float32, shape=[batch_size, xydim, xydim, cdim])
X_target = tf.placeholder(tf.float32, shape=[batch_size, xydim, xydim, cdim])
Y_source = tf.placeholder(tf.float32, shape=[batch_size, n_classes])
Y_target =  tf.placeholder(tf.float32, shape=[batch_size, n_classes])

# Placeholders for lambdas
lambda_r = tf.placeholder(tf.float32)
lambda_s = tf.placeholder(tf.float32)
lambda_p = tf.placeholder(tf.float32)
opt_lr = tf.placeholder(tf.float32)

# Intialize source/residual streams
lenet_model = LeNet(only_features=True, feature_dim=feature_dim).build(X_source)
resid_model = ResidualNet().build(lenet_model)

''' Weights for the classifier and discriminator '''
''' Weights for the classifier and discriminator '''
wC = {
    "classifier": tf.get_variable('classifier', shape=[500, n_classes], initializer=tf.contrib.layers.xavier_initializer())
}

wD = {
    "fc1": tf.get_variable('fc1_disc', shape=[n_classes, 500], initializer=tf.contrib.layers.xavier_initializer()),
    "fc2": tf.get_variable('fc2_disc', shape=[500, 500], initializer=tf.contrib.layers.xavier_initializer()),
    "logits": tf.get_variable('logits_disc', shape=[500, 2], initializer=tf.contrib.layers.xavier_initializer())
}


def classifier_logits(features):
    ''' Performs the classification on a feature vector '''
    return  tf.matmul(features, wC['classifier'])

def discriminator_logits(features):
    ''' Discrimnates between source and target features '''
    model = dense_relu(features, wD['fc1'])
    model = dropout(model, 0.5)
    model = dense_relu(model, wD['fc2'])
    model = dropout(model, 0.5)
    model = dense(model, wD['logits'])
    return model

# Compute A1/A2
update_A = resid_model.update_A(opt_lr, lambda_p)

# Get the source and target features/logits
source_features = lenet_model.stream(X_source)
target_features = lenet_model.stream(X_target, resid_model.get_target_weights())
source_logits = classifier_logits(source_features)
target_logits = classifier_logits(target_features)
source_disc_logits = discriminator_logits(source_logits)
target_disc_logits = discriminator_logits(target_logits)

# Loss for discriminator
target_label_disc = tf.concat([tf.zeros([batch_size, 1]), tf.ones([batch_size, 1])], axis=1)
source_label_disc = tf.concat([tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])], axis=1)
disc_loss_source = tf.nn.softmax_cross_entropy_with_logits(labels=source_label_disc, logits=source_disc_logits)
disc_loss_target = tf.nn.softmax_cross_entropy_with_logits(labels=target_label_disc, logits=target_disc_logits)
disc_loss = tf.reduce_mean(disc_loss_source) + tf.reduce_mean(disc_loss_target)

# Discrepency loss
class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_source, logits=source_logits))
disc_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=source_label_disc, logits=target_disc_logits))
disc_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_label_disc, logits=source_disc_logits))
conf_loss = lambda_r*(disc_loss_1 + disc_loss_2)

# Classification loss
class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_source, logits=source_logits))

# Residual loss
resid_loss = lambda_s*resid_model.get_resid_loss()

# Total loss
loss = class_loss + resid_loss + conf_loss

# Accuracy for classifier
pred_source = tf.argmax(tf.nn.softmax(source_logits), 1)
pred_target = tf.argmax(tf.nn.softmax(target_logits), 1)
acc_class_source = tf_accuracy(tf.argmax(Y_source, 1), pred_source)
acc_class_target = tf_accuracy(tf.argmax(Y_target, 1), pred_target)

# Accuracy for dsicriminator
pred_disc_source = tf.argmax(tf.nn.softmax(source_disc_logits), 1)
pred_disc_target = tf.argmax(tf.nn.softmax(target_disc_logits), 1)
acc_disc_source = tf_accuracy(tf.argmax(source_label_disc,1), pred_disc_source)
acc_disc_target = tf_accuracy(tf.argmax(target_label_disc,1), pred_disc_target)
acc_disc = (acc_disc_source+acc_disc_source)/2

# Initilize parameters and optimizer
opt_disc = tf.train.RMSPropOptimizer(learning_rate=1e-3, momentum=0.0)
opt_src = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)
disc_optimizer = opt_disc.minimize(disc_loss, var_list=list(wD.values()))
optimizer = opt_src.minimize(loss, var_list=lenet_model.get_weights()+list(wC.values())+resid_model.get_weights())

session = tf.Session()
session.run(tf.global_variables_initializer())

# Load the pretrained source stream
saver = tf.train.Saver(var_list=lenet_model.get_weights()+list(wC.values()))
if os.path.exists(checkpoint_dir+'/pretrained_svhn.ckpt.meta'):
    saver.restore(session, checkpoint_dir+'/pretrained_svhn.ckpt')
    log('[Info] Loaded pretrained source model.')

def test_on_complete_dataset():
    ''' Returns the training accuracy on all the training set '''
    batch_number=0
    training_accuracy=0
    while batch_number < 1800//batch_size+1:
        x_batch_target, y_batch_target = minibatch_svnh_mnist(batch_size, batch_number=batch_number)
        curr_training_acc =  session.run(acc_class_target, feed_dict={X_target: x_batch_target, Y_target:y_batch_target})
        batch_number +=1
        training_accuracy += curr_training_acc
        log("[Info] Testing on batch number {}, acc: {:.3}".format(batch_number, curr_training_acc))
    log("[Info] Training accuracy on the complete dataset: {:.3}".format(training_accuracy/batch_number))

    
print("[Info] Training the source stream and discriminator")
# 2. Training source stream and discriminator
l_r, l_s = 0.01, 0.1
max_iters, max_iters_adam = 1000, 1
for it in range(max_iters):
    # Train adam on a number of epochs
    for it_adam in range(max_iters_adam):
        # Load a minibatch
        x_batch_source, y_batch_source, x_batch_target, y_batch_target = minibatch_svnh_mnist(batch_size)

        # Do one step of optimization for discriminator and for the rest
        _, _, class_loss_curr, disc_loss_curr, loss_curr = session.run([disc_optimizer, optimizer, class_loss, disc_loss, loss], feed_dict={X_source:x_batch_source, Y_source:y_batch_source, X_target:x_batch_target, lambda_r: l_r, lambda_s: l_s})
    
    # Compute A1/A2
    optlr = session.run(opt_src._learning_rate_tensor)
    session.run(update_A, feed_dict={lambda_p:1, opt_lr:optlr})
    
    # Compute the training/validation accuracies for 
    src_tr_class_acc, tgt_tr_class_acc, src_tr_disc_acc, tgt_tr_disc_acc  = \
        session.run([acc_class_source, acc_class_target, acc_disc_source, acc_disc_target], feed_dict=
                    {X_source:x_batch_source, 
                     Y_source:y_batch_source, 
                     X_target: x_batch_target,
                     Y_target:y_batch_target})

    # Test after training
    log("[Info] Iteration {}, disc_loss: {:.4}, src_tr_class_acc: {:.3}, tgt_tr_class_acc: {:.3}, tr_disc_acc:{:.3}".format(it, disc_loss_curr, src_tr_class_acc, tgt_tr_class_acc, (src_tr_disc_acc+tgt_tr_disc_acc)/2))
        
    if it%100==0 and it > 1:
        test_on_complete_dataset()
        
# Compute the final rank of the target weights
target_weights = resid_model.get_weights(keys=True)
print("------- Target weights rank -------")
for k, w in target_weights.items():
    weight = session.run(w)
    print(k+"\t", w.get_shape().as_list(), np.linalg.matrix_rank(weight))

# Training accuracy after training
test_on_complete_dataset()


# def train_discriminator(max_iters=1, verbose=False):
#     if verbose:
#         print("[Info] Training the discriminator")
#     for it in range(max_iters):
        
#         # Load a minibatch
#         x_batch_source, _, x_batch_target, _ = minibatch_svnh_mnist(batch_size)

#         # Optimize one step for discriminator
#         _, disc_loss_curr, tr_disc_acc = session.run([disc_optimizer, disc_loss, acc_disc_target], 
#                                                          feed_dict={X_source:x_batch_source,
#                                                                     X_target:x_batch_target})
#         if verbose and it%10==0:
#             log("[Info] Iteration {}, disc_loss: {:.3}, disc_acc: {:.3}".format(it, disc_loss_curr, tr_disc_acc))
            
# Training accuracy before training
# test_on_complete_dataset()

# 1. Pretrain discriminator
# train_discriminator(100, True)
