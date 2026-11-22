"""
Generates samples based on a trained model (use LSGAN_v2.py to train, before using this 
file) based on second architecture of paper [1], for SVHN.

[1] Xudong Mao, Qing Liy, Haoran Xiez, Raymond Y. K. Laux, Zhen Wang, and Stephen Paul 
    Smolley. Least squares generative adversarial networks. arXiv:1611.04076v3 [cs.CV], 2017.
    
author: Bernardo Foresti (please cite)
"""
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as spio

def sample_z(m, n):
    '''
    Creates a mxn random matrix with data between -1 and +1 
    '''
    return np.random.uniform(-1., 1., size=[m, n]).astype(np.float32)

class create_y(object):
    '''
    Function that creates an one-hot vector representing the targeted labels to be created 
    by GAN Generator Net (G) model.
    '''
    def __init__(self, mb_size):
        self.mb = mb_size
        self.classes = np.array(range(10))
        self.n = self.classes.shape[0]
          
    def all_class(self):
        '''
        Given a mb_size, creates int(mb_size/n) one-hot labels tags
        '''        
        small_mb = int(self.mb/self.n)
        
        y = np.empty(shape=(1,1))
        for i in range(self.n):
            y_temp = self.classes[i]*np.ones(shape=(small_mb,1))
            y = np.concatenate((y,y_temp),axis=0)
        y = y[1:,:]
        lb = LabelBinarizer()
        y = np.array(lb.fit_transform(y))
        return y

def plot_imgs(samples, n_images, tag):
    '''
    Saves a frame of n_images x n_images
    '''   
    fig = plt.figure(figsize=(n_images, n_images))
    fig.suptitle("LSGAN Final, number:"+str(tag))
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
        
    filename = 'LSGAN_v2_tag'+str(tag)+'.png'
    plt.savefig(filename)
    plt.close()    
    return fig

def create(file_name):
    '''
    Function that loads the data
    '''
    dataset = spio.loadmat(file_name, squeeze_me=True)
    X_tr = np.transpose(dataset['X'], axes=(3,0,1,2))
    y_tr = dataset['y']
    X_tr = X_tr/127.5 - 1.
    return X_tr,y_tr

# End of functions -----------------------------------------------------------------------

# Loads dataset
X_tr_full,y_tr_full = create('test_32x32.mat')

# Uses just a part of dataset
ratio_tr = int(0.2*X_tr_full.shape[0])

del X_tr_full, y_tr_full

n_inputs_G = 64
n_classes = 10

tag = 'mixed'
y_in = create_y(ratio_tr).all_class()

mb_size = y_in.shape[0]
z_in = sample_z(mb_size,n_inputs_G)

# GAN Architecture -----------------------------------------------------------------------

# With this command, kernel doesn't have to be restarted each time routine is ran
tf.reset_default_graph()

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

# Creating samples from Generator: -------------------------------------------------------         

saver = tf.train.Saver()

# Creates data from trained model
with tf.Session() as sess:
    saver.restore(sess,'./LSGAN_v2.ckpt')
    G_new = G.eval(feed_dict={y: y_in, z: z_in})
    G_new_shuffle = np.random.permutation(G_new) 
    plot_imgs(G_new_shuffle,5,tag)
    
# Uncomment to save generated samples    
#np.save('X_model',G_new)
#np.save('y_model',y_in)

# End of code ----------------------------------------------------------------------------
    
    
    
