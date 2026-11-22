"""
Implementation of LSGAN as second architecture of paper [1], but for SVHN.
This architecture, after training, allows generation of specified class.

[1] Xudong Mao, Qing Liy, Haoran Xiez, Raymond Y. K. Laux, Zhen Wang, and Stephen Paul 
    Smolley. Least squares generative adversarial networks. arXiv:1611.04076v3 [cs.CV], 2017.
    
author: Bernardo Foresti (please cite)
"""
import scipy.io as spio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from sklearn.preprocessing import LabelBinarizer
import tensorflow.contrib.slim as slim

def leaky_relu(z, alpha=0.2):
    return tf.maximum(alpha*z, z)

def sample_z(m, n):
    '''
    Creates a mxn random matrix with data between -1 and +1 
    '''
    return np.random.uniform(-1., 1., size=[m, n]).astype(np.float32)

def create(file_name):
    '''
    Function that loads the data
    '''
    dataset = spio.loadmat(file_name, squeeze_me=True)
    X_tr = np.transpose(dataset['X'], axes=(3,0,1,2))
    y_tr = dataset['y']
    X_tr = X_tr/127.5 - 1.

    return X_tr,y_tr

def fetch_data(data_X, data_y, batch_index, batch_size):
    '''
    Function that creates batches for a given 'data'
    '''
    begin = batch_index*batch_size 
    end = (batch_index+1)*batch_size
    data_X_batch = data_X[begin:end,:,:,:]
    data_y_batch = data_y[begin:end,:]
    
    return data_X_batch,data_y_batch

def save_imgs(samples, n_images, step):
    '''
    Saves a frame of n_images x n_images
    '''   
    fig = plt.figure(figsize=(n_images, n_images))
    fig.suptitle("LSGAN_v2 step: " + str(step))
    gs = gridspec.GridSpec(n_images, n_images)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(n_images**2):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        samples[i] = (samples[i]+1.)
        samples[i] *= 127.5
        plt.imshow(np.clip(samples[i], 0, 255).astype(np.uint8))
        
    filename = 'LSGAN_v2_step'+str(step)+'.png'
    plt.savefig(filename)
    plt.close()
    
    return fig

# End of functions -----------------------------------------------------------------------

# Pre-processing:
# Loads dataset
X_tr_full,y_tr_full = create('test_32x32.mat')

# Uses just a part of dataset
ratio_tr = int(0.2*X_tr_full.shape[0])

# Creates X_tr and y_tr, y_tr as an array with shape=(m x n_classes) 
X_tr_sc = X_tr_full[0:ratio_tr,:,:,:]
y_tr = y_tr_full[0:ratio_tr]
lb = LabelBinarizer()
y_tr = np.array(lb.fit_transform(y_tr))

del X_tr_full,y_tr_full

# NN input dimensions
m,width,height,channels = X_tr_sc.shape
n_inputs_G = 64                          # Shape of random input
n_classes = y_tr.shape[1]                # Number of classes

# Minimization options
learning_rate_D = 0.001
learning_rate_G = 0.001
n_steps = 10000
mb_size = 32           # Mini batch size
n_batches = int(np.floor(m/mb_size))
print_steps = 10

# Initial step, if resuming a run, this is the last saved step + 1
resume_step = 0

# GAN Architecture -----------------------------------------------------------------------

# With this command, kernel doesn't have to be restarted each time routine is ran
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(mb_size, width, height, channels), name="x")
y = tf.placeholder(tf.float32, shape=(mb_size, n_classes), name="y")
z = tf.placeholder(tf.float32, shape=(mb_size, n_inputs_G), name="z")

with tf.variable_scope("G_net"):
    # 1st Concatenate
    input_G = tf.concat([z,y],axis=1) 
    # 1st hidden layer
    hidden1_G = slim.fully_connected(input_G, 8*8*128, normalizer_fn=slim.batch_norm,
                                     activation_fn=tf.nn.relu, scope='hidden1')
    hidden1_G_reshape = tf.reshape(hidden1_G,shape=[mb_size,8,8,128])
    # 2nd hidden layer
    hidden2_G = slim.convolution2d_transpose(hidden1_G_reshape, 128, 5, stride=2, 
                                             normalizer_fn=slim.batch_norm, 
                                             activation_fn=tf.nn.relu, scope='hidden2')
    # Output layer
    G = slim.convolution2d_transpose(hidden2_G, 3, 5, stride=2,
                                     activation_fn=tf.nn.tanh, scope='output')                                     

with tf.variable_scope("D_net"):
    # One-hot 1st layer
    y_reshape_D1 = tf.reshape(y,shape=(mb_size,1,1,n_classes))    
    # 1st hidden layer
    cl1_D1 = slim.convolution(x, 256, 5, stride=2, activation_fn=tf.identity, scope='hidden1')
    cl1_D1_leaky = leaky_relu(cl1_D1)
    # 1st Concatenate    
    ones_1_D1 = tf.ones(shape=(mb_size,16,16,n_classes))
    y_ones1_D1 = y_reshape_D1*ones_1_D1
    input1_D1 = tf.concat([cl1_D1_leaky,y_ones1_D1],axis=3)
    # 2nd hidden layer
    cl2_D1 = slim.convolution(input1_D1, 320, 5, stride=2, normalizer_fn=slim.batch_norm,
                              activation_fn=tf.identity, scope='hidden2')
    cl2_D1_leaky = leaky_relu(cl2_D1)
    # 2nd Concatenate
    ones_2_D1 = tf.ones(shape=(mb_size,8,8,n_classes))
    y_ones2_D1 = y_reshape_D1*ones_2_D1
    input2_D1 = tf.concat([cl2_D1_leaky,y_ones2_D1],axis=3)
    
    # 3rd hidden layer
    flatten_D1 = tf.contrib.layers.flatten(input2_D1)
    hidden3_D1 = slim.fully_connected(flatten_D1, 1024, normalizer_fn=slim.batch_norm,
                                      activation_fn=tf.identity, scope='hidden3')
    bn3_D1_leaky = leaky_relu(hidden3_D1)
    # 3rd Concatenate
    input3_D1 = tf.concat([bn3_D1_leaky,y],axis=1)    
    # Output layer
    D1 = slim.fully_connected(input3_D1, 1, activation_fn=tf.identity, scope='output')


with tf.variable_scope("D_net", reuse=True):
    # One-hot 1st layer
    y_reshape_D2 = tf.reshape(y,shape=(mb_size,1,1,n_classes))    
    # 1st hidden layer
    cl1_D2 = slim.convolution(G, 256, 5, stride=2, activation_fn=tf.identity, scope='hidden1')
    cl1_D2_leaky = leaky_relu(cl1_D2)
    # 1st Concatenate    
    ones_1_D2 = tf.ones(shape=(mb_size,16,16,n_classes))
    y_ones1_D2 = y_reshape_D2*ones_1_D2
    input1_D2 = tf.concat([cl1_D2_leaky,y_ones1_D2],axis=3)
    # 2nd hidden layer
    cl2_D2 = slim.convolution(input1_D2, 320, 5, stride=2, normalizer_fn=slim.batch_norm,
                              activation_fn=tf.identity, scope='hidden2')
    cl2_D2_leaky = leaky_relu(cl2_D2)
    # 2nd Concatenate
    ones_2_D2 = tf.ones(shape=(mb_size,8,8,n_classes))
    y_ones2_D2 = y_reshape_D2*ones_2_D2
    input2_D2 = tf.concat([cl2_D2_leaky,y_ones2_D2],axis=3)
    
    # 3rd hidden layer
    flatten_D2 = tf.contrib.layers.flatten(input2_D2)
    hidden3_D2 = slim.fully_connected(flatten_D2, 1024, normalizer_fn=slim.batch_norm,
                                      activation_fn=tf.identity, scope='hidden3')
    bn3_D2_leaky = leaky_relu(hidden3_D2)
    # 3rd Concatenate
    input3_D2 = tf.concat([bn3_D2_leaky,y],axis=1)    
    # Output layer
    D2 = slim.fully_connected(input3_D2, 1, activation_fn=tf.identity, scope='output')

# Minimization objectives
with tf.name_scope("loss"):
    # Determines the Loss functions, following LSGAN definition
    loss_D = 0.5 * (tf.reduce_mean((D1 - 1)**2) + tf.reduce_mean(D2**2))
    loss_G = 0.5 * tf.reduce_mean((D2 - 1)**2)

# Selects variables for training
vars = tf.trainable_variables()
var_D_net = [v for v in vars if v.name.startswith('D_net/')]            
var_G_net = [v for v in vars if v.name.startswith('G_net/')]

with tf.name_scope("train"):
    optimizer_D = tf.train.AdamOptimizer(learning_rate_D, 0.5)
    training_op_D = optimizer_D.minimize(loss_D, var_list=var_D_net)
    optimizer_G = tf.train.AdamOptimizer(learning_rate_G, 0.5)
    training_op_G = optimizer_G.minimize(loss_G, var_list=var_G_net)

with tf.name_scope('eval'):
    out = tf.placeholder(tf.float32, shape=(None, 1), name="prediction")
    acc = tf.reduce_mean(out)

# Begin training: ------------------------------------------------------------------------

init_glob = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:

    # Check if it's a resume run (inside if) or a new run (inside else)
    if os.path.isfile('./LSGAN_v2.ckpt.meta'):
        saver.restore(sess, './LSGAN_v2.ckpt')
    else:
        init_glob.run()
    
    for step in range(resume_step, n_steps):
        print('step:',step)
        np.random.seed(step) 
        img = np.random.permutation(X_tr_sc)
        np.random.seed(step) 
        y_img = np.random.permutation(y_tr)
        
        for batch_index in range(n_batches):
            print('batch:',batch_index,'of:',n_batches)
            # Train Discriminator: -------------------------------------------------------
            x_in, y_in = fetch_data(img, y_img, batch_index, mb_size)
            z_in = sample_z(mb_size,n_inputs_G)
            _,loss_D_it = sess.run([training_op_D,loss_D],
                                   feed_dict={x: x_in, y: y_in, z: z_in})
            # Train Generator: -----------------------------------------------------------
            z_in = sample_z(mb_size,n_inputs_G)
            _,loss_G_it = sess.run([training_op_G,loss_G],
                                   feed_dict={y: y_in, z: z_in})
            
        # Print loss each 'print_steps' steps
        if step % print_steps == 0:
            G_new = G.eval(feed_dict={y: y_in, z: z_in})
            D_new = D1.eval(feed_dict={x: G_new, y: y_in})
            accuracy = acc.eval(feed_dict={out: D_new})
            save_imgs(G_new, 5, step)            
            print ("%d [D loss: %f, mean_D: %.2f%%] [G loss: %f]" 
                   % (step, loss_D_it, 100*accuracy, loss_G_it))
            save_path = saver.save(sess, "./LSGAN_v2.ckpt")

# End of code ----------------------------------------------------------------------------    
