import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.preprocessing import OneHotEncoder
from random import shuffle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from random import shuffle

######################################################################
######################### 2D Conv Layers #############################
######################################################################
# Convolutional layers
def conv2d(X, W, strides=[1,1,1,1], padding='SAME', bias=None):
    conv = tf.nn.conv2d(X, W, strides=strides, padding=padding)
    conv = conv if(bias==None)else tf.nn.bias_add(conv,bias)
    return conv

def conv2d_sigmoid(X, W, strides=[1,1,1,1], padding='SAME', bias=None):
    return tf.nn.sigmoid(conv2d(X, W, strides=strides, padding=padding, bias=bias))

def conv2d_relu(X, W, strides=[1,1,1,1], padding='SAME', bias=None):
    return tf.nn.relu(conv2d(X, W, strides=strides, padding=padding, bias=bias))

def conv2d_lrelu(X, W, strides=[1,1,1,1], padding='SAME', bias=None):
    return lrelu(conv2d(X, W, strides=strides, padding=padding, bias=bias))

def conv2d_tanh(X,W,strides=[1,1,1,1],padding='SAME', bias=None):
    return tf.nn.tanh(conv2d(X, W, strides=strides, padding=padding, bias=bias))

def max_pool2d(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
    return tf.nn.max_pool(input, ksize=ksize, padding=padding, strides=strides)

def avg_pool2d(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
    return tf.nn.avg_pool(input, ksize=ksize, padding=padding, strides=strides, data_format='NHWC')

def dropout(input, ratio):
    return tf.nn.dropout(input, ratio)


## Fully connected layer 
def dense(input, W, bias=None):
    dense = tf.matmul(input, W)
    if bias != None:
        dense = bias+dense
    return dense

def dense_relu(input, W, bias=None):
    return tf.nn.relu(dense(input, W, bias))

def dense_sigmoid(input, W, bias=None):
    return tf.nn.sigmoid(dense(input, W, bias))

####################### Helper functions ########################
def xavier_init(size):
    in_dim = size[-1]
    ou_dim = size[-2]
    k_size = size[1]
    xavier_stddev = tf.sqrt(2.0/(in_dim*ou_dim*k_size)) 
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def random_uniform(size, min_=0, max_=1):
    return tf.random_uniform(size, minval=min_, maxval=max_)
    
def deconv_shape(input_shape, filter, s=2):
    k,_,d,_ = filter.get_shape().as_list()
    h,w = input_shape[0], input_shape[1]
    output_shape = [(h-1)*s+k, (w-1)*s+k, d]
    return output_shape

 ######################## Data Loader ########################
def minibatch_usps_mnist(batch_size, type_='train', test_size=0.1, batch_number=None):
    source_images = np.load('./dataset/mnist_usps/usps_1800_ims.npy')
    target_images = np.load('./dataset/mnist_usps/mnist_2000_ims.npy')

    source_labels = np.load('./dataset/mnist_usps/usps_1800_labels.npy')
    target_labels = np.load('./dataset/mnist_usps/mnist_2000_labels.npy')

    source_ids = np.load('./dataset/mnist_usps/usps_1800_ids.npy')
    target_ids = np.load('./dataset/mnist_usps/mnist_2000_ids.npy')
    
    idx_cut_source = int((1-test_size)*len(source_ids))
    idx_cut_target = int((1-test_size)*len(target_ids))
    
    if not batch_number is None:
        start_id = (batch_number*batch_size)%(len(target_ids)-batch_size)
        end_id = start_id+batch_size
        X_target = target_images[target_ids[start_id:end_id]].reshape(-1, 16, 16, 1).astype(np.float64)
        Y_target = toOneHot(target_labels[target_ids[start_id:end_id]], n_classes=10)
        return X_target, Y_target
    
    if type_ == 'train':
        source_ids = source_ids[:idx_cut_source]
        target_ids = target_ids[:idx_cut_target]
    else:
        source_ids = source_ids[idx_cut_source:]
        target_ids = target_ids[idx_cut_target:]

    np.random.shuffle(source_ids)
    np.random.shuffle(target_ids)

    X_source = source_images[source_ids[:batch_size]].reshape(-1, 16, 16, 1).astype(np.float64)
    X_target = target_images[target_ids[:batch_size]].reshape(-1, 16, 16, 1).astype(np.float64)
    Y_source = toOneHot(source_labels[source_ids[:batch_size]], n_classes=10)
    Y_target = toOneHot(target_labels[target_ids[:batch_size]], n_classes=10)

    return X_source, Y_source, X_target, Y_target

def minibatch_svnh_mnist(batch_size, type_='train', test_size=0.1, batch_number=None):
    source_images = np.load('./dataset/mnist_svnh/train_svnh_images.npy')
    target_images = np.load('./dataset/mnist_svnh/train_mnist_images.npy')

    source_labels = np.load('./dataset/mnist_svnh/train_svnh_labels.npy')
    target_labels = np.load('./dataset/mnist_svnh/train_mnist_labels.npy')

    source_ids = np.load('./dataset/mnist_svnh/svnh_73k_ids.npy')
    target_ids = np.load('./dataset/mnist_svnh/mnist_60k_ids.npy')
    
    idx_cut_source = int((1-test_size)*len(source_ids))
    idx_cut_target = int((1-test_size)*len(target_ids))
    
    if not batch_number is None:
        start_id = (batch_number*batch_size)%(len(target_ids)-batch_size)
        end_id = start_id+batch_size
        X_target = target_images[target_ids[start_id:end_id]].reshape(-1, 28, 28, 1).astype(np.float64)
        Y_target = toOneHot(target_labels[target_ids[start_id:end_id]], n_classes=10)
        return X_target, Y_target
    
    if type_ == 'train':
        source_ids = source_ids[:idx_cut_source]
        target_ids = target_ids[:idx_cut_target]
    else:
        source_ids = source_ids[idx_cut_source:]
        target_ids = target_ids[idx_cut_target:]

    np.random.shuffle(source_ids)
    np.random.shuffle(target_ids)

    X_source = source_images[source_ids[:batch_size]].reshape(-1, 28, 28, 1).astype(np.float64)
    X_target = target_images[target_ids[:batch_size]].reshape(-1, 28, 28, 1).astype(np.float64)
    Y_source = toOneHot(source_labels[source_ids[:batch_size]], n_classes=10)
    Y_target = toOneHot(target_labels[target_ids[:batch_size]], n_classes=10)

    return X_source, Y_source, X_target, Y_target


def minibatch_tiny_imagenet(batch_size, type_='train'):
    home_dir="datasets/tiny-imagenet-200/"
    
    ''' TODO: REMOVE NOVEL CLASSES AFTER VGG16 WORKS '''
    with open(os.path.join(home_dir, "wnids.txt")) as lf:
        list_classes = [a for a in sorted(lf.read().split("\n")) if len(a)>2]

    with open(os.path.join(home_dir, "list_{}.txt".format(type_))) as ltf:
        list_images = [a for a in ltf.read().split("\n") if len(a)>2]
    shuffle(list_images)
    X, y = [], []
    for file_path in list_images[:batch_size]:
        if type_ == 'test':
            file_path, synset = file_path.split("\t")
        else:
            synset = file_path.split("/")[-3]
        file_path = os.path.join(home_dir, file_path)
        
        img = cv2.imread(file_path)
        label = list_classes.index(synset)
        
        X.append(img)
        y.append(label)
    
    y = np.array(toOneHot(np.array(y).reshape(-1, 1), len(list_classes)))
    X = np.array(X)
    return X, y

def minibatch_caltech(batch_size, type_='train', class_='source', img_size=None):
    home_dir="datasets/101_ObjectCategories/"
    
    with open(os.path.join(home_dir, "{}_classes.txt".format(class_))) as lf:
        list_classes = sorted([a for a in sorted(lf.read().split("\n")) if len(a)>2])

    with open(os.path.join(home_dir, "{}_{}.txt".format(type_, class_))) as ltf:
        list_images = [a for a in ltf.read().split("\n") if len(a)>2]
        
    shuffle(list_images)
    X, y = [], []
    for file_path in list_images[:batch_size]:
        file_path, label = file_path.split("\t")
        file_path = os.path.join(home_dir, file_path)
        
        img = cv2.imread(file_path)
        if not img_size is None:
            img=cv2.resize(img, img_size)
        label = int(label)
        
        X.append(img)
        y.append(label)
    
    y = np.array(toOneHot(np.array(y).reshape(-1, 1), len(list_classes)))
    X = np.array(X)
    return X, y

# Other helper functions
def toOneHot(vector, n_classes=None):
    enc = OneHotEncoder(n_values=n_classes)
    enc = enc.fit(vector)
    return enc.transform(vector).toarray()

def tf_accuracy(y_true, y_pred):
    equality = tf.equal(y_true, y_pred)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy

def topkaccuracy(y_true, y_pred, k=1):
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=y_pred, targets=y_true, k=k), tf.float32))

def l2_loss_list(tensors):
    l2_loss = None
    for t in tensors:
        if l2_loss is None:
            l2_loss = tf.nn.l2_loss(t)
        else:
            l2_loss = l2_loss+tf.nn.l2_loss(t)
    return l2_loss

def log(line, log_file='log.txt'):
    print(line)
    if not os.path.exists('./log/'):
        os.mkdir('./log/')
    with open(os.path.join('./log/', log_file), 'a') as lf:
        lf.write(line+'\n')
        
def clear_log():
    os.remove('./log/log.txt')
    
def convert_image(img):
    img = (img-img.min())/(img.max()-img.min())
    img *= 255
    return img.astype(np.uint8)

def count_lines(file_path):
    with open(file_path) as of:
        return len(of.read().split("\n"))
    
def latest_model(folder_path):
    list_files=[]
    for path, subdirs, files in os.walk(folder_path):
        for name in sorted(files, reverse=True, key=lambda name: os.path.getmtime(os.path.join(path, name))):
            if len(name.split(".")) == 3 and name.split(".")[1]=='ckpt':
                return ".".join(name.split(".")[:2])
    return None

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_curve(x, fname, xaxis='', yaxis='', mv_avg=True):
    plt.plot(x)
    plt.plot(moving_average(x,n=10))
    if xaxis != '':
        plt.xlabel(xaxis)
    if yaxis != '':
        plt.ylabel(yaxis)
    plt.savefig(fname)

def shuffle_both(a,b):
    ids = list(range(len(a)))
    shuffle(ids)
    return a[ids], b[ids]