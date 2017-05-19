from tensorflow.contrib import layers
from mayavi import mlab

import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class VarShapeLearner(object):
    def __init__(self,
                 obj_res,
                 batch_size,
                 global_latent_dim,
                 local_latent_dim,
                 local_latent_num
                 ):
        # define model parameters
        self.obj_res = obj_res
        self.batch_size = batch_size
        self.global_latent_dim = global_latent_dim
        self.local_latent_dim = local_latent_dim
        self.local_latent_num = local_latent_num

        # define input placeholder
        self.input_shape = [self.batch_size] + [self.obj_res]*3 + [1]
        self.x = tf.placeholder(tf.float32, self.input_shape)
        self.y = tf.placeholder(tf.float32, [self.batch_size, 100, 100, 3])
        self.gamma = tf.placeholder(tf.float32, shape=[])
        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        # create model and define its loss and optimizer
        self._model_create()
        self._model_loss_optimizer()

        # start tensorflow session
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # initialize model weights (using dictionary style)
    def _weights_init(self):
        # z_0, z_i parameters
        self.z_mean, self.z_logstd, self.z_all, self.kl_loss = ([0]*(self.local_latent_num+1) for _ in range(4))

        # z_0 -> z_i parameters
        self.enc_zzi_fclayer1, self.enc_zzi_fclayer2 = [[0]*self.local_latent_num for _ in range(2)]

        # z_i -> z_{i+1} parameters
        self.enc_zizi_fclayer1, self.enc_zizi_fclayer2 = [[0]*(self.local_latent_num-1) for _ in range(2)]

        # all -> z_i parameters
        self.enc_allzi_fclayer1, self.enc_allzi_fclayer2 = ([0]*self.local_latent_num for _ in range(2))

        # x -> z_0, z_i conv layers
        self.enc_conv1, self.enc_conv2, self.enc_conv3 = ([0]*(self.local_latent_num+1) for _ in range(3))
        self.enc_fclayer1, self.enc_fclayer2 = ([0]*(self.local_latent_num+1) for _ in range(2))

        self.weights_all = dict()
        self.weights_all['W'] = {
            # input_shape x -> all_lat z, z_i (0, 1:local_lat_num)
            'enc_conv1': [tf.get_variable(name='enc_conv1', shape=[6, 6, 6, 1, 32],
                                          initializer=layers.xavier_initializer())]*(self.local_latent_num+1),
            'enc_conv2': [tf.get_variable(name='enc_conv2', shape=[5, 5, 5, 32, 64],
                                          initializer=layers.xavier_initializer())]*(self.local_latent_num+1),
            'enc_conv3': [tf.get_variable(name='enc_conv3', shape=[4, 4, 4, 64, 128],
                                          initializer=layers.xavier_initializer())]*(self.local_latent_num+1),
            'enc_fc1'  : [tf.get_variable(name='enc_fc1', shape=[1024, 256],
                                          initializer=layers.xavier_initializer())]*(self.local_latent_num+1),
            'enc_fc2'  : [tf.get_variable(name='enc_fc2', shape=[256, 100],
                                          initializer=layers.xavier_initializer())]*(self.local_latent_num+1),

            # global_lat z_0 -> local_lat z_i
            'zzi_fc1' : [tf.get_variable(name='zzi_fc1', shape=[self.global_latent_dim, 100],
                                         initializer=layers.xavier_initializer())]*self.local_latent_num,
            'zzi_fc2' : [tf.get_variable(name='zzi_fc2', shape=[100, 100],
                                         initializer=layers.xavier_initializer())]*self.local_latent_num,

            # local_lat z_i -> local_lat z_{i+1}
            'zizi_fc1': [tf.get_variable(name='zizi_fc1', shape=[self.local_latent_dim, 100],
                                         initializer=layers.xavier_initializer())]*(self.local_latent_num-1),
            'zizi_fc2': [tf.get_variable(name='zizi_fc2', shape=[100, 100],
                                         initializer=layers.xavier_initializer())]*(self.local_latent_num-1),

            # input_shape x -> global_lat z_0
            'z_mean'  : tf.get_variable(name='z_mean', shape=[100, self.global_latent_dim],
                                        initializer=layers.xavier_initializer()),
            'z_logstd': tf.get_variable(name='z_logstd', shape=[100, self.global_latent_dim],
                                        initializer=layers.xavier_initializer()),

            # combined [x, z_i, z_0] -> local_lat z_{i+1} (i >= 1)
            'allzi_fc1':[tf.get_variable(name='allz1_fc1', shape=[200, 100],
                                         initializer=layers.xavier_initializer())]+
                        [tf.get_variable(name='allzi_fc1', shape=[300, 100],
                                         initializer=layers.xavier_initializer())]*(self.local_latent_num-1),
            'allzi_fc2': [tf.get_variable(name='allzi_fc2', shape=[100, 100],
                                          initializer=layers.xavier_initializer())]*self.local_latent_num,
            'zi_mean'  : [tf.get_variable(name='zi_mean', shape=[100, self.local_latent_dim],
                                          initializer=layers.xavier_initializer())]*self.local_latent_num,
            'zi_logstd': [tf.get_variable(name='zi_logstd', shape=[100, self.local_latent_dim],
                                          initializer=layers.xavier_initializer())]*self.local_latent_num,

            # combined lat [z_0, z_i] - > input_shape x
            'dec_fc1'  : tf.get_variable(name='dec_zfc1',
                                         shape=[self.global_latent_dim+self.local_latent_num*self.local_latent_dim,
                                                100 * (self.local_latent_num + 1)],
                                          initializer=layers.xavier_initializer()),
            'dec_fc2'  : tf.get_variable(name='dec_fc2', shape=[100 * (self.local_latent_num + 1), 1024],
                                         initializer=layers.xavier_initializer()),
            'dec_conv1': tf.get_variable(name='dec_conv1', shape=[4, 4, 4, 64, 128],
                                          initializer=layers.xavier_initializer()),
            'dec_conv2': tf.get_variable(name='dec_conv2', shape=[5, 5, 5, 32, 64],
                                          initializer=layers.xavier_initializer()),
            'dec_conv3': tf.get_variable(name='dec_conv3', shape=[6, 6, 6, 1, 32],
                                          initializer=layers.xavier_initializer()),

            # image decoder Im -> z'
            'image_conv1': tf.get_variable(name='image_conv1', shape=[32, 32, 3, 16],
                                            initializer=layers.xavier_initializer()),
            'image_conv2': tf.get_variable(name='image_conv2', shape=[15, 15, 16, 32],
                                          initializer=layers.xavier_initializer()),
            'image_conv3': tf.get_variable(name='image_conv3', shape=[5, 5, 32, 64],
                                            initializer=layers.xavier_initializer()),
            'image_conv4': tf.get_variable(name='image_conv4', shape=[3, 3, 64, 128],
                                            initializer=layers.xavier_initializer()),
            'image_fc1': tf.get_variable(name='image_fc1', shape=[512, 200],
                                        initializer=layers.xavier_initializer()),
            'image_fc2': tf.get_variable(name='image_fc2', shape=[200, self.global_latent_dim
                                                                  + self.local_latent_num * self.local_latent_dim],
                                        initializer=layers.xavier_initializer()),
        }

        self.weights_all['b'] = {
            # input_shape x -> all_lat z_0, z_i (0, 1:local_lat_num)
            'enc_conv1' : [tf.Variable(name='enc_conv1', initial_value=tf.zeros(32))]*(self.local_latent_num+1),
            'enc_conv2' : [tf.Variable(name='enc_conv2', initial_value=tf.zeros(64))]*(self.local_latent_num+1),
            'enc_conv3' : [tf.Variable(name='enc_conv3', initial_value=tf.zeros(128))]*(self.local_latent_num+1),
            'enc_fc1'   : [tf.Variable(name='enc_fc1', initial_value=tf.zeros(256))]*(self.local_latent_num+1),
            'enc_fc2'   : [tf.Variable(name='enc_fc2', initial_value=tf.zeros(100))]*(self.local_latent_num+1),

            # global_lat z_0 -> local_lat z_i
            'zzi_fc1': [tf.Variable(name='zzi_fc1', initial_value=tf.zeros(100))] * self.local_latent_num,
            'zzi_fc2': [tf.Variable(name='zzi_fc2', initial_value=tf.zeros(100))] * self.local_latent_num,

            # local_lat z_i -> local_lat z_{i+1}
            'zizi_fc1': [tf.Variable(name='zizi_fc1', initial_value=tf.zeros(100))] * (self.local_latent_num-1),
            'zizi_fc2': [tf.Variable(name='zizi_fc2', initial_value=tf.zeros(100))] * (self.local_latent_num-1),

            # input_shape x -> global_lat z_0
            'z_mean'  : tf.Variable(name='z_mean', initial_value=tf.zeros(self.global_latent_dim)),
            'z_logstd': tf.Variable(name='z_logstd', initial_value=tf.zeros(self.global_latent_dim)),

            # combined [x, z_i, z_0] -> local_lat z_{i+1} (i >= 1)
            'allzi_fc1': [tf.Variable(name='allzi_fc1', initial_value=tf.zeros(100))]*self.local_latent_num,
            'allzi_fc2': [tf.Variable(name='allzi_fc2', initial_value=tf.zeros(100))]*self.local_latent_num,
            'zi_mean'  : [tf.Variable(name='zi_mean', initial_value=tf.zeros(self.local_latent_dim))] * self.local_latent_num,
            'zi_logstd': [tf.Variable(name='zi_logstd', initial_value=tf.zeros(self.local_latent_dim))] * self.local_latent_num,

            # combined lat [z, z_i] - > input_shape x
            'dec_fc1': tf.Variable(name='dec_fc1', initial_value=tf.zeros(100*(self.local_latent_num + 1))),
            'dec_fc2': tf.Variable(name='dec_fc2', initial_value=tf.zeros(1024)),
            'dec_conv1': tf.Variable(name='dec_conv1', initial_value=tf.zeros(64)),
            'dec_conv2': tf.Variable(name='dec_conv2', initial_value=tf.zeros(32)),
            'dec_conv3': tf.Variable(name='dec_conv3', initial_value=tf.zeros(1)),

            # image reconstruction
            'image_conv1': tf.Variable(name='enc_conv1', initial_value=tf.zeros(16)),
            'image_conv2': tf.Variable(name='enc_conv2', initial_value=tf.zeros(32)),
            'image_conv3': tf.Variable(name='enc_conv3', initial_value=tf.zeros(64)),
            'image_conv4': tf.Variable(name='enc_conv3', initial_value=tf.zeros(128)),
            'image_fc1': tf.Variable(name='image_fc1', initial_value=tf.zeros(200)),
            'image_fc2': tf.Variable(name='image_fc2', initial_value=tf.zeros(self.global_latent_dim
                                                                  + self.local_latent_num * self.local_latent_dim)),
        }

    # use re-parametrization trick
    def _sampling(self, z_mean, z_logstd, latent_dim):
        epsilon = tf.random_normal((self.batch_size, latent_dim))
        return z_mean + tf.exp(z_logstd) * epsilon

    # define inference model q(z_0:n|x)
    def _inf_model(self, weights, biases):
        # input_shape x -> local_lat z_i
        for i in range(self.local_latent_num + 1):
            self.enc_conv1[i] = tf.nn.relu(tf.nn.conv3d(self.x, weights['enc_conv1'][i],
                                                        strides=[1, 2, 2, 2, 1], padding='VALID')
                                                        + biases['enc_conv1'][i])
            self.enc_conv2[i] = tf.nn.relu(tf.nn.conv3d(self.enc_conv1[i], weights['enc_conv2'][i],
                                                        strides=[1, 2, 2, 2, 1], padding='VALID')
                                                        + biases['enc_conv2'][i])
            self.enc_conv3[i] = tf.nn.relu(tf.nn.conv3d(self.enc_conv2[i], weights['enc_conv3'][i],
                                                        strides=[1, 1, 1, 1, 1], padding='VALID')
                                                        + biases['enc_conv3'][i])
            self.enc_conv3[i] = tf.reshape(self.enc_conv3[i], [self.batch_size, 1024])
            self.enc_fclayer1[i] = tf.nn.relu(tf.matmul(self.enc_conv3[i], weights['enc_fc1'][i])
                                              + biases['enc_fc1'][i])
            self.enc_fclayer2[i] = tf.nn.relu(tf.matmul(self.enc_fclayer1[i], weights['enc_fc2'][i])
                                              + biases['enc_fc2'][i])
        # sample global latent variable
        self.z_mean[0] = tf.matmul(self.enc_fclayer2[0], weights['z_mean']) + biases['z_mean']
        self.z_logstd[0] = tf.matmul(self.enc_fclayer2[0], weights['z_logstd']) + biases['z_logstd']
        self.z_all[0] = self._sampling(self.z_mean[0], self.z_logstd[0], self.global_latent_dim)

        for i in range(self.local_latent_num):
            # z -> z_i
            self.enc_zzi_fclayer1[i] = tf.nn.relu(tf.matmul(self.z_all[0], weights['zzi_fc1'][i])
                                                  + biases['zzi_fc1'][i])
            self.enc_zzi_fclayer2[i] = tf.nn.relu(tf.matmul(self.enc_zzi_fclayer1[i], weights['zzi_fc2'][i])
                                                  + biases['zzi_fc2'][i])

            if i == 0:  # sampling z_1
                self.enc_allzi_fclayer1[i] = tf.nn.relu(tf.matmul(tf.concat([self.enc_zzi_fclayer2[i], self.enc_fclayer2[i+1]], axis=1),
                                                                  weights['allzi_fc1'][i]) + biases['allzi_fc1'][i])
                self.enc_allzi_fclayer2[i] = tf.nn.relu(tf.matmul(self.enc_zzi_fclayer1[i],
                                                                  weights['allzi_fc2'][i]) + biases['allzi_fc2'][i])

                self.z_mean[1] = tf.matmul(self.enc_allzi_fclayer2[i], weights['zi_mean'][i]) + biases['zi_mean'][i]
                self.z_logstd[1] = tf.matmul(self.enc_allzi_fclayer2[i], weights['zi_logstd'][i]) + biases['zi_logstd'][i]
                self.z_all[1] = self._sampling(self.z_mean[1], self.z_logstd[1],self.local_latent_dim)
            else:   # sampling z_i (i >= 1)
                self.enc_zizi_fclayer1[i-1] = tf.nn.relu(tf.matmul(self.z_all[i], weights['zizi_fc1'][i-1])
                                                      + biases['zizi_fc1'][i-1])
                self.enc_zizi_fclayer2[i-1] = tf.nn.relu(tf.matmul(self.enc_zizi_fclayer1[i-1], weights['zizi_fc2'][i-1])
                                                      + biases['zizi_fc2'][i-1])
                self.enc_allzi_fclayer1[i] = tf.nn.relu(tf.matmul(tf.concat([self.enc_zzi_fclayer2[i], self.enc_fclayer2[i+1], self.enc_zizi_fclayer2[i-1]], axis=1),
                                                                  weights['allzi_fc1'][i]) + biases['allzi_fc1'][i])
                self.enc_allzi_fclayer2[i] = tf.nn.relu(tf.matmul(self.enc_allzi_fclayer1[i], weights['allzi_fc2'][i])
                                                        + biases['allzi_fc2'][i])
                self.z_mean[i+1] = tf.matmul(self.enc_allzi_fclayer2[i], weights['zi_mean'][i]) + biases['zi_mean'][i]
                self.z_logstd[i+1] = tf.matmul(self.enc_allzi_fclayer2[i], weights['zi_logstd'][i]) + biases['zi_logstd'][i]
                self.z_all[i+1] = self._sampling(self.z_mean[i+1], self.z_logstd[i+1], self.local_latent_dim)

        # concat latent codes for shape classification
        self.latent_feature = tf.concat([self.z_mean[i] for i in range(self.local_latent_num + 1)], axis=1)

    # define generative model p(x|z_0:n)
    def _gen_model(self, weights, biases):
        dec_fclayer1 = tf.nn.relu(tf.matmul(self.latent_feature , weights['dec_fc1']) + biases['dec_fc1'])
        dec_fclayer2 = tf.nn.relu(tf.matmul(dec_fclayer1 , weights['dec_fc2']) + biases['dec_fc2'])
        dec_fclayer2 = tf.reshape(dec_fclayer2, [self.batch_size, 2, 2, 2, 128])
        dec_conv1    = tf.nn.relu(tf.nn.conv3d_transpose(dec_fclayer2, weights['dec_conv1'],
                                  output_shape=[self.batch_size, 5, 5, 5, 64],
                                  strides=[1, 1, 1, 1, 1],padding='VALID') + biases['dec_conv1'])
        dec_conv2    = tf.nn.relu(tf.nn.conv3d_transpose(dec_conv1, weights['dec_conv2'],
                                  output_shape=[self.batch_size, 13, 13, 13, 32],
                                  strides=[1, 2, 2, 2, 1], padding='VALID') + biases['dec_conv2'])
        dec_conv3    = tf.nn.sigmoid(tf.nn.conv3d_transpose(dec_conv2, weights['dec_conv3'],
                                     output_shape=[self.batch_size, 30, 30, 30, 1],
                                     strides=[1, 2, 2, 2, 1], padding='VALID') + biases['dec_conv3'])
        return dec_conv3

    # define image decoder
    def _image_decoder(self, weights, biases):
        image_conv1 = tf.nn.relu(tf.nn.conv2d(self.y, weights['image_conv1'],
                                              strides=[1, 2, 2, 1], padding='VALID')
                                              + biases['image_conv1'])
        image_conv2 = tf.nn.relu(tf.nn.conv2d(image_conv1, weights['image_conv2'],
                                              strides=[1, 2, 2, 1], padding='VALID')
                                              + biases['image_conv2'])
        image_conv3 = tf.nn.relu(tf.nn.conv2d(image_conv2, weights['image_conv3'],
                                              strides=[1, 2, 2, 1], padding='VALID')
                                              + biases['image_conv3'])
        image_conv4 = tf.nn.relu(tf.nn.conv2d(image_conv3, weights['image_conv4'],
                                              strides=[1, 1, 1, 1], padding='VALID')
                                              + biases['image_conv4'])
        image_conv4 = tf.reshape(image_conv4, [self.batch_size, 512])
        image_fclayer1 = tf.nn.relu(tf.matmul(image_conv4, weights['image_fc1']) + biases['image_fc1'])
        image_fclayer1 = tf.nn.dropout(image_fclayer1, keep_prob=self.keep_prob)
        image_fclayer2 = tf.matmul(image_fclayer1, weights['image_fc2'] + biases['image_fc2'])
        return image_fclayer2


    # create model
    def _model_create(self):
        # load defined network structure
        self._weights_init()
        network_weights = self.weights_all

        # learn gaussian parameters from inference network
        self._inf_model(network_weights['W'], network_weights['b'])

        # reconstruct training data from sampled latent states
        self.x_rec = self._gen_model(network_weights['W'], network_weights['b'])

        # reconstruct latent code
        self.learned_feature = self._image_decoder(network_weights['W'], network_weights['b'])

    # define VSL loss and optimizer
    def _model_loss_optimizer(self):
        # define reconstruction loss (binary cross-entropy)
        self.rec_loss = -tf.reduce_sum(self.x * tf.log(1e-5 + self.x_rec)
                                       +(1-self.x) * tf.log(1e-5 + 1 - self.x_rec), axis=1)

        # define kl loss KL(p(z|x)||q(z))
        for i in range(self.local_latent_num + 1):
            self.kl_loss[i] = -0.5 * tf.reduce_sum(1 + 2 * self.z_logstd[i] - tf.square(self.z_mean[i]) - tf.square(tf.exp(self.z_logstd[i])), axis=1)

        self.kl_loss_all = tf.add_n(self.kl_loss)

        # latent image decoder loss
        self.lat_loss = tf.nn.l2_loss(tf.abs(self.latent_feature - self.learned_feature))

        # total loss = kl loss + rec loss + lat loss
        self.loss = tf.reduce_mean(self.rec_loss + 0.001 * self.kl_loss_all + self.gamma * self.lat_loss)

        # gradient clipping to avoid nan
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-5)
        gradients = optimizer.compute_gradients(self.loss)
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)
        clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
        self.optimizer = optimizer.apply_gradients(clipped_gradients)

    # train model on mini-batch
    def model_fit(self, x, y, gamma, keep_prob):
        opt, cost = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: x, self.y: y,
                                                                          self.gamma: gamma, self.keep_prob: keep_prob})
        return cost


# define network structure, parameters
global_latent_dim  = 5
local_latent_dim   = 2
local_latent_num   = 3
obj_res     = 30
batch_size  = 5
print_step  = 2
total_epoch = 10

# 3D visualization
def draw_sample(voxel,  savepath):
    voxel = np.reshape(voxel, (obj_res, obj_res, obj_res))
    xx, yy, zz = np.where(voxel >= 0)
    ss = voxel[np.where(voxel >= 0)] * 1.
    mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(400, 400))
    s = mlab.points3d(xx, yy, zz, ss,
                      mode="cube",
                      colormap='bone',
                      scale_factor=2)
    mlab.view(112, 242, 80)
    s.scene.light_manager.lights[0].activate  = True
    s.scene.light_manager.lights[0].intensity = 1.0
    s.scene.light_manager.lights[0].elevation = 50
    s.scene.light_manager.lights[0].azimuth   = -30
    s.scene.light_manager.lights[1].activate  = True
    s.scene.light_manager.lights[1].intensity = 0.3
    s.scene.light_manager.lights[1].elevation = -40
    s.scene.light_manager.lights[1].azimuth   = -30
    s.scene.light_manager.lights[2].activate  = False
    if savepath == 0:
        return mlab.show()
    return  mlab.savefig(savepath)


# load dataset (only PASCAL3D in this case)
PASCAL = h5py.File('dataset/PASCAL3D.mat')
image_train = np.transpose(PASCAL['image_train'])
model_train = np.transpose(PASCAL['model_train'])
image_test = np.transpose(PASCAL['image_test'])
model_test = np.transpose(PASCAL['model_test'])

# load VSL model
VSL = VarShapeLearner(obj_res=obj_res,
                      batch_size=batch_size,
                      global_latent_dim=global_latent_dim,
                      local_latent_dim=local_latent_dim,
                      local_latent_num=local_latent_num)

# load saved parameters here, comment this to train model from scratch.
#VSL.saver.restore(VSL.sess, os.path.abspath('parameters/your_model_name.ckpt'))

def unison_shuffled_copies(a, b):
    '''solution using: 
    http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison/4602224'''
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def IOU(a,b):
    index = np.argwhere(a == 1)
    index_gt = np.argwhere(b == 1)
    intersect = np.intersect1d(index, index_gt)
    union = np.union1d(index, index_gt)
    IOU = len(intersect) / len(union)
    return IOU

# training VSL model
name_list = ['aero', 'bike', 'boat', 'bus', 'car', 'chair', 'mbike', 'sofa', 'train', 'tv']

id = 1 # training separately per class, using id from the name_list [1:10]
test_indx  = np.where(model_test[:,0] == id)
modelid_test = model_test[test_indx[0], 1:]
imageid_test = image_test[test_indx[0],:]

train_indx  = np.where(model_train[:,0] == id)
modelid_train = model_train[train_indx[0], 1:]
imageid_train = image_train[train_indx[0],:]

for epoch in range(total_epoch):
    cost     = np.zeros(4, dtype=np.float32)
    avg_cost = np.zeros(4, dtype=np.float32)
    train_batch = int(imageid_train.shape[0] / batch_size)

    index = epoch + 0  # correct the training index, set 0 for training from scratch

    # randomly shuffle for each epoch
    [imageid_train, modelid_train] = unison_shuffled_copies(imageid_train, modelid_train)

    # warming-up schedule
    if index <= 50:
        gamma = 10 ** (np.floor(index / 10) - 8)
    elif 50 < index < 100:
        gamma = np.floor((index - 40) / 10) * 10 ** (-3)
    else:
        gamma = 5 * 10 ** (-3)

    # iterate for all batches
    for i in range(train_batch):
        x_train = modelid_train[batch_size*i:batch_size*(i+1),:].reshape([batch_size, obj_res, obj_res, obj_res, 1])
        y_train = imageid_train[batch_size*i:batch_size*(i+1),:]

        # calculate and average kl, rec and latent loss for each batch
        cost[0] = np.mean(VSL.sess.run(VSL.kl_loss_all, feed_dict={VSL.x: x_train, VSL.y: y_train,
                                                                   VSL.gamma: gamma, VSL.keep_prob: 0.2}))
        cost[1] = np.mean(VSL.sess.run(VSL.rec_loss, feed_dict={VSL.x: x_train, VSL.y: y_train,
                                                                VSL.gamma: gamma, VSL.keep_prob: 0.2}))
        cost[2] = np.mean(VSL.sess.run(VSL.lat_loss, feed_dict={VSL.x: x_train, VSL.y: y_train,
                                                                VSL.gamma: gamma, VSL.keep_prob: 0.2}))
        cost[3] = VSL.model_fit(x_train, y_train, gamma, 0.2)

        avg_cost += cost / train_batch

    print("Epoch: {:04d} | kl-loss: {:.4f} + rec-loss: {:.4f} + lat_loss: {:.4f} = total-loss: {:.4f}"
          .format(index, avg_cost[0], avg_cost[1], avg_cost[2], avg_cost[3]))

    if index % print_step == 0:
        draw_sample(VSL.sess.run(VSL.x_rec[0,:], feed_dict={VSL.x: x_train, VSL.y: y_train, VSL.gamma: gamma}), 'plots/rec-%d.png' % index)
        mlab.close()

        VSL.saver.save(VSL.sess, os.path.abspath('parameters/{}-{:03d}-3-2-5-cost-{:.4f}.ckpt'
                                                 .format(name_list[id-1], index, avg_cost[3])))

# IOU training and testing results
test_batch = int(modelid_test.shape[0] / batch_size)
z_train = [[0]]*test_batch
z_test = [[0]]*test_batch
for i in range(test_batch):
    x_train = modelid_train[batch_size * i:batch_size * (i + 1), :].reshape([batch_size, obj_res, obj_res, obj_res, 1])
    y_train = imageid_train[batch_size * i:batch_size * (i + 1), :]
    x_test = modelid_test[batch_size * i:batch_size * (i + 1), :].reshape([batch_size, obj_res, obj_res, obj_res, 1])
    y_test = imageid_test[batch_size * i:batch_size * (i + 1), :]

    z_train[i] = VSL.sess.run(VSL.learned_feature, feed_dict={VSL.x: x_train, VSL.y: y_train, VSL.gamma: gamma, VSL.keep_prob:1})
    z_test[i] = VSL.sess.run(VSL.learned_feature, feed_dict={VSL.x: x_test, VSL.y: y_test, VSL.gamma: gamma, VSL.keep_prob:1})

    if i == 0:
        train_rec = VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_train[i]})
        test_rec = VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_test[i]})
    else:
        train_rec = np.concatenate((train_rec,  VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_train[i]})))
        test_rec = np.concatenate((test_rec,  VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_test[i]})))

train_rec = np.floor(train_rec + 0.5)
train_rec = train_rec.reshape(len(train_rec), obj_res ** 3)

test_rec = np.floor(test_rec + 0.5)
test_rec = test_rec.reshape(len(test_rec), obj_res ** 3)

prob_train = 0
prob_test = 0
test_batch = int(modelid_test.shape[0] / batch_size)
for i in range(batch_size * test_batch):
    prob_model = IOU(train_rec[i, :], modelid_train[i, :])
    prob_train = prob_model / (batch_size * test_batch) + prob_train

    prob_model = IOU(test_rec[i, :], modelid_test[i, :])
    prob_test = prob_model / (batch_size * test_batch) + prob_test

print('IOU - {} - Train: {:.4f}, Test: {:.4f}'.format(name_list[id-1], prob_train, prob_test))


# image reconstruction
test_indx  = np.where(model_test[:,0] == id)
modelid_test = model_test[test_indx[0], 1:]
imageid_test = image_test[test_indx[0],:]
test_batch = int(modelid_test.shape[0] / batch_size)
gamma = 5e-3
z_learned = [[0]]*test_batch
for i in range(test_batch):
    x_train = modelid_test[batch_size * i:batch_size * (i + 1), :].reshape([batch_size, obj_res, obj_res, obj_res, 1])
    y_train = imageid_test[batch_size * i:batch_size * (i + 1), :]

    z_learned[i] = VSL.sess.run(VSL.learned_feature, feed_dict={VSL.x: x_train, VSL.y: y_train, VSL.gamma: gamma, VSL.keep_prob:1})
    if i == 0:
        A = VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_learned[i]})
    else:
        A = np.concatenate((A,  VSL.sess.run(VSL.x_rec, feed_dict={VSL.latent_feature: z_learned[i]})))

A = np.floor(A + 0.5)
A = A.reshape(len(A), obj_res ** 3)

# plot image and its 3d reconstructed model
for i in range(20):
    plt.imshow(imageid_test[i, :])
    plt.axis('off')
    plt.savefig('plots/{}-im{:d}.png'.format(name_list[id-1], i))
    plt.close()
    draw_sample(A[i, :], 'plots/{}-md{:d}.png'.format(name_list[id-1], i))
    mlab.close()


