from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os, pdb, random
import keras
from keras.models import Sequential
from keras.layers.core import *
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras.applications.mobilenet import MobileNet
import dvs128_aedat_reader as aeread
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y, d):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))

def identity_triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss 
        
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
      
    Returns:
      the triplet loss as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
            
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
          
    return loss

def create_pairs(x, digit_indices, classes):
    """ Positive and negative pair creation.
        Alternates between positive and negative pairs.
    """
   # pdb.set_trace()
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(classes)]) - 1
    for d in range(classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, classes)
            dn = (d + inc) % classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(in_dim):
    """ Base network to be shared (eq. to feature extraction).
    """
    seq = Sequential()
    seq.add(Dense(128, input_shape=(in_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq

def create_conv_network(num_classes=4, in_shape=[10,28,28]):
    
    input = Input(shape=in_shape)
    x = Conv2D(64, kernel_size=(3, 3),
                     activation='relu')(input)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(num_classes, activation='softmax')(x)
    #pdb.set_trace()
    return Model(input,x)

def compute_accuracy(predictions, labels):
    """ Compute classification accuracy with a fixed threshold on distances.
    """
    return labels[predictions.ravel() < 0.5].mean()


def pair(X_train, X_test, y_train, y_test, classes): 
    # create training+test positive and negative pairs
    classes = 4
    y_train = y_train-1
    y_test = y_test-1
    digit_indices = [np.where(y_train == i)[0] for i in range(classes)]
    tr_pairs, tr_y = create_pairs(X_train, digit_indices, classes)

    digit_indices = [np.where(y_test == i)[0] for i in range(classes)]
    te_pairs, te_y = create_pairs(X_test, digit_indices, classes)
    return tr_pairs, te_pairs, tr_y, te_y

#Should pop out with X being of shape (num_samples, time_steps, 28, 28) and Y shape (numsamples) in range classes
filepath2 = '/Users/twelsh/Neuromorph2018/meltpool_data/multiwell_fixture/DVS128-2016-09-24T22-13-51-0600-0293-0_12_Hz_gallium.aedat'
filepath1 = '/Users/twelsh/Neuromorph2018/meltpool_data/multiwell_fixture/DVS128-2016-09-24T22-_gaussian_gallium.aedat'
(X_train, y_train), (X_test, y_test) = aeread.get_aed_frames(filepath1, events_per_frame = 1000, num_frames = 1000, hop_ratio = 2, time_steps=10)
(X_train2, y_train2), (X_test2, y_test2) = aeread.get_aed_frames(filepath2, events_per_frame = 1000, num_frames = 1000, hop_ratio = 2, time_steps=10)
#pdb.set_trace()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train2 = X_train2.astype('float32')
X_test2 = X_test2.astype('float32')
classes = 4
in_dim = [10,28,28]
nb_epoch = 200 
tr_pairs, te_pairs, tr_y, te_y = pair(X_train, X_test, y_train, y_test, classes)
tr_pairs2, te_pairs2, tr_y2, te_y2 = pair(X_train2, X_test2, y_train2, y_test2, classes)
#pdb.set_trace()
# network definition
# create a Sequential for each element of the pairs

# share base network with both inputs
# G_w(input1), G_w(input2) in article
base_network = create_conv_network(classes,in_dim)
input_a = Input(shape=in_dim)
input_b = Input(shape=in_dim)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)
# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
#make a data generator for augmentation
datagen = ImageDataGenerator(
  featurewise_center=False,
  featurewise_std_normalization=False,
  rotation_range=20,
  width_shift_range=0.2,
  height_shift_range=0.2,
  horizontal_flip=True)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

do_train = 0
if do_train:
    datagen.fit(X_train)

    # fits the model on batches with real-time data augmentation:
    tb = keras.callbacks.TensorBoard(log_dir='./tb_logs', histogram_freq=0,  
              write_graph=True, write_images=True)
    model.fit_generator(datagen.flow([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=64),
                        steps_per_epoch=len(tr_pairs[:,0]) / 64, epochs=nb_epoch, callbacks = [tb])
    model.save('last_model.h5')
    #model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=128, nb_epoch=nb_epoch,
    #          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
else:
   model.load_weights('last_model.h5')
# compute final accuracy on training and test sets
pred1 = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred1, tr_y)
pred2 = model.predict([te_pairs2[:, 0], te_pairs2[:, 1]])
te_acc = compute_accuracy(pred2, te_y2)

#from read_activations import get_activations
#model.summary()
#pdb.set_trace()
#doop = get_activations(model,[te_pairs2[:, 0], te_pairs2[:, 1]])
#print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
#print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
no_lamb = Model(inputs = model.input,outputs=model.layers[-2].get_output_at(1))
pred1 = no_lamb.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
pred2 = no_lamb.predict([te_pairs[:, 0], te_pairs[:, 1]])
from sklearn.manifold import TSNE
x_embed = TSNE(n_components=3).fit_transform(np.concatenate((pred1,pred2),axis=0))
pdb.set_trace()
fig = plt.figure()
fig.add_subplot(111, projection='3d')
plt.scatter(x_embed[0:len(pred1),0],x_embed[0:len(pred1),1],x_embed[0:len(pred1),2],c='r')
plt.scatter(x_embed[len(pred1):,0],x_embed[len(pred1):,1],x_embed[len(pred1):,2],c='b')
plt.show()
pdb.set_trace()
