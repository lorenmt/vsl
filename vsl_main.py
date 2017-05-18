from tensorflow.contrib import layers
from sklearn import svm, manifold
from mayavi import mlab

import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class VarShapeLearner(object):
    def __init__(self, obj_res,
                 batch_size,
                 global_latent_dim,
                 local_latent_dim,
                 local_latent_num):
        # define model parameters
        self.obj_res = obj_res
        self.batch_size = batch_size
        self.global_latent_dim = global_latent_dim
        self.local_latent_dim = local_latent_dim
        self.local_latent_num = local_latent_num

        # define input placeholder
        self.input_shape = [self.batch_size] + [self.obj_res]*3 + [1]
        self.x = tf.placeholder(tf.float32, self.input_shape)

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
        self.z_mean, self.z_logstd, self.z_all, self.kl_loss = ([0] * (self.local_latent_num + 1) for _ in range(4))

        # z_0 -> z_i parameters
        self.enc_zzi_fclayer1, self.enc_zzi_fclayer2 = [[0] * self.local_latent_num for _ in range(2)]

        # z_i -> z_{i+1} parameters
        self.enc_zizi_fclayer1, self.enc_zizi_fclayer2 = [[0] * (self.local_latent_num - 1) for _ in range(2)]

        # all -> z_i parameters
        self.enc_allzi_fclayer1, self.enc_allzi_fclayer2 = ([0] * self.local_latent_num for _ in range(2))

        # x -> z_0, z_i conv layers
        self.enc_conv1, self.enc_conv2, self.enc_conv3 = ([0] * (self.local_latent_num + 1) for _ in range(3))
        self.enc_fclayer1, self.enc_fclayer2 = ([0] * (self.local_latent_num + 1) for _ in range(2))

        # define all weights and biases of the network
        self.weights_all = dict()
        self.weights_all['W'] = {
            # input_shape x -> all_lat z_0, z_i (0, 1:local_lat_num)
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

            # input_shape x -> global_lat z
            'z_mean'  : tf.get_variable(name='z_mean', shape=[100, self.global_latent_dim],
                                        initializer=layers.xavier_initializer()),
            'z_logstd': tf.get_variable(name='z_logstd', shape=[100, self.global_latent_dim],
                                        initializer=layers.xavier_initializer()),

            # merged [x, z_i, z_0] -> local_lat z_{i+1} (i >= 1)
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
                                          initializer=layers.xavier_initializer())
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

            # local_lat z_i -> local_lat z_{i+1}
            'z_mean'  : tf.Variable(name='z_mean', initial_value=tf.zeros(self.global_latent_dim)),
            'z_logstd': tf.Variable(name='z_logstd', initial_value=tf.zeros(self.global_latent_dim)),

            # combined [x, z_i, z_0] -> local_lat z_{i+1} (i >= 1)
            'allzi_fc1': [tf.Variable(name='allzi_fc1', initial_value=tf.zeros(100))]*self.local_latent_num,
            'allzi_fc2': [tf.Variable(name='allzi_fc2', initial_value=tf.zeros(100))] * self.local_latent_num,
            'zi_mean'  : [tf.Variable(name='zi_mean', initial_value=tf.zeros(self.local_latent_dim))] * self.local_latent_num,
            'zi_logstd': [tf.Variable(name='zi_logstd', initial_value=tf.zeros(self.local_latent_dim))] * self.local_latent_num,

            # combined lat [z_0, z_i] - > input_shape x
            'dec_fc1': tf.Variable(name='dec_fc1', initial_value=tf.zeros(100*(self.local_latent_num + 1))),
            'dec_fc2': tf.Variable(name='dec_fc2', initial_value=tf.zeros(1024)),
            'dec_conv1': tf.Variable(name='dec_conv1', initial_value=tf.zeros(64)),
            'dec_conv2': tf.Variable(name='dec_conv2', initial_value=tf.zeros(32)),
            'dec_conv3': tf.Variable(name='dec_conv3', initial_value=tf.zeros(1)),
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
            # z_0 -> z_i
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

        # concatenate all latent codes
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

    # create model
    def _model_create(self):
        # load defined network structure
        self._weights_init()
        network_weights = self.weights_all

        # learn latent parameters from inference network
        self._inf_model(network_weights['W'], network_weights['b'])

        # reconstruct training data from learned latent features
        self.x_rec = self._gen_model(network_weights['W'], network_weights['b'])

    # define VSL loss and optimizer
    def _model_loss_optimizer(self):
        # define reconstruction loss (binary cross-entropy)
        self.rec_loss = -tf.reduce_sum(self.x * tf.log(1e-5 + self.x_rec)
                                       +(1-self.x) * tf.log(1e-5 + 1 - self.x_rec), axis=1)

        # define kl loss
        for i in range(self.local_latent_num + 1):
            self.kl_loss[i] = -0.5 * tf.reduce_sum(1 + 2 * self.z_logstd[i] - tf.square(self.z_mean[i]) - tf.square(tf.exp(self.z_logstd[i])), axis=1)

        self.kl_loss_all = tf.add_n(self.kl_loss)

        # total loss = kl loss + rec loss
        self.loss = tf.reduce_mean(self.rec_loss + 0.001*self.kl_loss_all)

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
    def model_fit(self, x):
        opt, cost = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: x})
        return cost


# define network structure, parameters
global_latent_dim  = 20
local_latent_dim   = 10
local_latent_num   = 5
obj_res     = 30
batch_size  = 200
print_step  = 5
total_epoch = 500

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
    mlab.view(120, 290, 85)
    s.scene.light_manager.lights[0].activate  = True
    s.scene.light_manager.lights[0].intensity = 1.0
    s.scene.light_manager.lights[0].elevation = 30
    s.scene.light_manager.lights[0].azimuth   = -30
    s.scene.light_manager.lights[1].activate  = True
    s.scene.light_manager.lights[1].intensity = 0.3
    s.scene.light_manager.lights[1].elevation = -60
    s.scene.light_manager.lights[1].azimuth   = -30
    s.scene.light_manager.lights[2].activate  = False
    if savepath == 0:
        return mlab.show()
    return  mlab.savefig(savepath)

# load dataset (pick modelnet40 or modelnet10)
data = h5py.File('dataset/ModelNet40_res30_raw.mat')

train_all = np.transpose(data['train'])
test_all  = np.transpose(data['test'])


# load VSL model
VSL = VarShapeLearner(obj_res=obj_res,
                      batch_size=batch_size,
                      global_latent_dim=global_latent_dim,
                      local_latent_dim=local_latent_dim,
                      local_latent_num=local_latent_num)

# load saved parameters here, comment this to train model from scratch.
VSL.saver.restore(VSL.sess, os.path.abspath('parameters/modelnet40-2619-cost-1.1170.ckpt'))


# training VSL model
for epoch in range(total_epoch):
    cost     = np.zeros(3, dtype=np.float32)
    avg_cost = np.zeros(3, dtype=np.float32)
    train_batch = int(train_all.shape[0] / batch_size)

    index = epoch + 2620 # correct the training index, set 0 for training from scratch

    # iterate for all batches
    np.random.shuffle(train_all)
    for i in range(train_batch):
        x_train = train_all[batch_size*i:batch_size*(i+1),1:].reshape([batch_size, obj_res, obj_res, obj_res, 1])

        # calculate and average kl and vae loss for each batch
        cost[0] = np.mean(VSL.sess.run(VSL.kl_loss_all, feed_dict={VSL.x: x_train}))
        cost[1] = np.mean(VSL.sess.run(VSL.rec_loss, feed_dict={VSL.x: x_train}))
        cost[2] = VSL.model_fit(x_train)
        avg_cost += cost / train_batch

    print("Epoch: {:04d} | kl-loss: {:.4f} + rec-loss: {:.4f} = total-loss: {:.4f}"
          .format(index, avg_cost[0], avg_cost[1], avg_cost[2]))

    if epoch % print_step == 0:
        draw_sample(VSL.sess.run(VSL.x_rec[0,:], feed_dict={VSL.x: x_train}), 'plots/rec-%d.png' % index)
        mlab.close()

        VSL.saver.save(VSL.sess, os.path.abspath('parameters/modelnet40-{:04d}-cost-{:.4f}.ckpt'.format(index, avg_cost[2])))


# shape classification using SVM
'''note: this process will concatenate all features in the dataset
which will be needed for tsne output.'''

train_batch = int(train_all.shape[0] / batch_size)
np.random.shuffle(train_all)
for i in range(train_batch):
    x_train = train_all[batch_size * i:batch_size * (i + 1), 1:].reshape([batch_size, obj_res, obj_res, obj_res, 1])
    if i == 0:
        train_feature = VSL.sess.run(VSL.latent_feature, feed_dict={VSL. x:x_train})
    else:
        train_feature = np.concatenate([train_feature, VSL.sess.run(VSL.latent_feature, feed_dict={VSL. x:x_train})])

np.random.shuffle(test_all)
test_batch = int(test_all.shape[0] / batch_size)
for i in range(test_batch):
    x_test = test_all[batch_size * i:batch_size * (i + 1), 1:].reshape(
        [batch_size, obj_res, obj_res, obj_res, 1])
    if i == 0:
        test_feature = VSL.sess.run(VSL.latent_feature, feed_dict={VSL.x: x_test})
    else:
        test_feature = np.concatenate(
            [test_feature, VSL.sess.run(VSL.latent_feature, feed_dict={VSL.x: x_test})])

clf = svm.SVC(kernel='rbf')
clf.fit(train_feature[:,:], train_all[0:batch_size * train_batch, 0])

train_accuracy = np.sum(train_all[0:batch_size * train_batch, 0] == clf.predict(train_feature[:,:])) / (train_batch * batch_size)
test_accuracy  = np.sum(test_all[0:batch_size * test_batch, 0] == clf.predict(test_feature[:,:])) / (test_batch * batch_size)

print('Shape classification: train: {:.4f}, test: {:.4f}'
      .format(train_accuracy, test_accuracy))


# t-sne 2D visualization
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(train_feature)
ax = plt.figure(figsize=(8, 8), facecolor='white')
plt.scatter(Y[:, 0], Y[:, 1], c=train_all[0:batch_size * train_batch, 0], edgecolors='none', cmap='terrain')
plt.xticks([])
plt.yticks([])
plt.axis('tight')
plt.show()
plt.savefig('plot/tsne.png')


# shape generation with Gaussian noise
'''please fetch model id number from ModelNet: http://modelnet.cs.princeton.edu/
ModelNet40 and ModelNet10 has 40 and 10 classes respectively with alphabetical naming order.
Here is the example id = 1 means with all model in "airplane" class. '''
id = 1
test_indx  = np.where(test_all[:,0] == id)
test  = test_all[test_indx[-0],1:]
x_test = test_all[test_indx[0][0]:test_indx[0][0]+batch_size,1:].reshape([batch_size, obj_res, obj_res, obj_res, 1])

z = VSL.sess.run(VSL.latent_feature, feed_dict={VSL.x: x_train})
z_new = z + np.random.normal(scale=0.02, size=[batch_size, ])
for i in range(20):
    draw_sample(VSL.sess.run(VSL.x_rec[i, :], feed_dict={VSL.latent_feature: z_new}), 'plots/rec_{:d}.png'.format(i))
    mlab.close()


# shape interpolation
'''shape interpolation visualization from two reconstructed shapes.
id1 and id2 means two shapes instances in the reconstructed shape batches.
note: id numbers cannot exceed the batch_size.'''
z = VSL.sess.run(VSL.latent_feature, feed_dict={VSL.x: x_test})
id1 = 40
id2 = 11
d = z[id1, :] - z[id2, :]
for i in range(7):
    draw_sample(VSL.sess.run(VSL.x_rec[id2, :], feed_dict={VSL.latent_feature: z}), 'plots/interpolation_conetoilet-{:d}.png'.format(i))
    mlab.close()
    z[id2, :] = z[id2, :] + d / 6

