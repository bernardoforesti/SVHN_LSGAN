"""
Implementation of LSGAN as first architecture of paper [1], but for SVHN.
This architecture, after training, does not allow generation of specified class.

[1] Xudong Mao, Qing Liy, Haoran Xiez, Raymond Y. K. Laux, Zhen Wang, and Stephen Paul 
    Smolley. Least squares generative adversarial networks. arXiv:1611.04076v3 [cs.CV], 2017.
    
author: Bernardo Foresti (please cite)
"""
import scipy.io as spio
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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
    data_y_batch = data_y[begin:end]
    return data_X_batch,data_y_batch

def save_imgs(samples, n_images, step):
    '''
    Saves a frame of n_images x n_images
    '''   
    fig = plt.figure(figsize=(n_images, n_images))
    fig.suptitle("LSGAN_v1 step: " + str(step))
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
        
    filename = 'LSGAN_v1_step'+str(step)+'.png'
    plt.savefig(filename)
    plt.close()
    return fig

# End of functions -----------------------------------------------------------------------

# Pre-processing:
# Loads dataset
X_tr_full,y_tr_full = create('test_32x32.mat')

# Uses just a part of dataset
ratio_tr = int(0.2*X_tr_full.shape[0])
X_tr_sc = X_tr_full[0:ratio_tr,:,:,:]
y_tr = y_tr_full[0:ratio_tr]

del X_tr_full,y_tr_full

# NN input dimensions
m,width,height,channels = X_tr_sc.shape
n_inputs_G = 64                          # Shape of random input
n_classes = 10                           # Number of classes

# Minimization options
learning_rate_D = 0.001
learning_rate_G = 0.001
n_steps = 10000
mb_size = 128           # Mini batch size
n_batches = int(np.floor(m/mb_size))
print_steps = 10

# Initial step, if resuming a run, this is the last saved step
resume_step = 0

# GAN Architecture -----------------------------------------------------------------------

# With this command, kernel doesn't have to be restarted each time routine is ran
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(mb_size, width, height, channels), name="x")
y = tf.placeholder(tf.float32, shape=(mb_size, n_classes), name="y")
z = tf.placeholder(tf.float32, shape=(mb_size, n_inputs_G), name="z")

with tf.variable_scope("G_net"):
    input_G = slim.fully_connected(z, 2*2*256, normalizer_fn=slim.batch_norm, 
                                   activation_fn=tf.identity, scope='input_G')
    input_G = tf.reshape(input_G, [mb_size, 2, 2, 256])
    
    conv1 = slim.convolution2d_transpose(input_G, 256, 3, stride=2, 
                                         normalizer_fn=slim.batch_norm, 
                                         activation_fn=tf.identity, scope='g_conv1')
    conv1 = tf.nn.relu(conv1)
    
    conv2 = slim.convolution2d_transpose(conv1, 256, 3, stride=1, 
                                         normalizer_fn=slim.batch_norm, 
                                         activation_fn=tf.identity, scope='g_conv2')
    conv2 = tf.nn.relu(conv2)
    
    conv3 = slim.convolution2d_transpose(conv2, 256, 3, stride=2, 
                                         normalizer_fn=slim.batch_norm, 
                                         activation_fn=tf.identity, scope='g_conv3')
    conv3 = tf.nn.relu(conv3)
    
    conv4 = slim.convolution2d_transpose(conv3, 256, 3, stride=1, 
                                         normalizer_fn=slim.batch_norm, 
                                         activation_fn=tf.identity, scope='g_conv4')
    conv4 = tf.nn.relu(conv4)
    
    conv5 = slim.convolution2d_transpose(conv4, 128, 3, stride=2, 
                                         normalizer_fn=slim.batch_norm, 
                                         activation_fn=tf.identity, scope='g_conv5')
    conv5 = tf.nn.relu(conv5)
    
    conv6 = slim.convolution2d_transpose(conv5, 64, 3, stride=2, 
                                         normalizer_fn=slim.batch_norm, 
                                         activation_fn=tf.identity, scope='g_conv6')
    conv6 = tf.nn.relu(conv6)
    
    conv7 = slim.convolution2d_transpose(conv6, 3, 3, stride=1, activation_fn=tf.identity, 
                                         scope='g_conv7')
    G = tf.nn.tanh(conv7)                            

with tf.variable_scope("D_net"):
    conv1 = slim.convolution(x, 64, 5, stride=2, activation_fn=tf.identity, scope='d_conv1')
    conv1 = leaky_relu(conv1)
    
    conv2 = slim.convolution(conv1, 128, 5, stride=2, normalizer_fn=slim.batch_norm, 
                             activation_fn=tf.identity, scope='d_conv2')
    conv2 = leaky_relu(conv2)
    
    conv3 = slim.convolution(conv2, 256, 5, stride=2, normalizer_fn=slim.batch_norm, 
                             activation_fn=tf.identity, scope='d_conv3')
    conv3 = leaky_relu(conv3)
    
    conv4 = slim.convolution(conv3, 512, 5, stride=2, normalizer_fn=slim.batch_norm, 
                             activation_fn=tf.identity, scope='d_conv4')
    conv4 = leaky_relu(conv4)    
    conv4 = tf.reshape(conv4, [mb_size, 2*2*512])
    
    D1 = slim.fully_connected(conv4, 1, scope='d_fc1', activation_fn=tf.identity)

with tf.variable_scope("D_net", reuse=True):
    conv1 = slim.convolution(G, 64, 5, stride=2, activation_fn=tf.identity, scope='d_conv1')
    conv1 = leaky_relu(conv1)
    
    conv2 = slim.convolution(conv1, 128, 5, stride=2, normalizer_fn=slim.batch_norm, 
                             activation_fn=tf.identity, scope='d_conv2')
    conv2 = leaky_relu(conv2)
    
    conv3 = slim.convolution(conv2, 256, 5, stride=2, normalizer_fn=slim.batch_norm, 
                             activation_fn=tf.identity, scope='d_conv3')
    conv3 = leaky_relu(conv3)
    
    conv4 = slim.convolution(conv3, 512, 5, stride=2, normalizer_fn=slim.batch_norm, 
                             activation_fn=tf.identity, scope='d_conv4')
    conv4 = leaky_relu(conv4)    
    conv4 = tf.reshape(conv4, [mb_size, 2*2*512])
    
    D2 = slim.fully_connected(conv4, 1, scope='d_fc1', activation_fn=tf.identity)
    
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
    if os.path.isfile('./LSGAN_v1.ckpt.meta'):
        saver.restore(sess, './LSGAN_v1.ckpt')
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
            x_in,_ = fetch_data(img, y_img, batch_index, mb_size)
            z_in = sample_z(mb_size,n_inputs_G)
            _,loss_D_it = sess.run([training_op_D,loss_D],
                                   feed_dict={x: x_in, z: z_in})
            # Train Generator: -----------------------------------------------------------
            z_in = sample_z(mb_size,n_inputs_G)
            _,loss_G_it = sess.run([training_op_G,loss_G],
                                   feed_dict={z: z_in})
            
        # Print loss each 'print_steps' steps
        if step % print_steps == 0:
            G_new = G.eval(feed_dict={z: z_in})
            D_new = D1.eval(feed_dict={x: G_new})
            accuracy = acc.eval(feed_dict={out: D_new})
            save_imgs(G_new, 5, step)            
            print ("%d [D loss: %f, mean_D: %.2f%%] [G loss: %f]" 
                   % (step, loss_D_it, 100*accuracy, loss_G_it))
            save_path = saver.save(sess, "./LSGAN_v1.ckpt")

# End of code ----------------------------------------------------------------------------