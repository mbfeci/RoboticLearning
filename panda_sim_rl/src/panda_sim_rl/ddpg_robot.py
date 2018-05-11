#!/usr/bin/env python
"""
Environment is a Robot Arm. The arm tries to get to the blue point.
The environment will return a geographic (distance) information for the arm to learn.

The far away from blue point the less reward; touch blue r+=1; stop at blue for a while then get r=+10.
 
You can train this RL by using LOAD = False, after training, this model will be store in the a local folder.
Using LOAD = True to reload the trained model for playing.

You can customize this script in a way you want.

View more on [Python] : https://morvanzhou.github.io/tutorials/

Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
tensorflow >= 1.0.1
"""

import tensorflow as tf
import numpy as np
import os
import shutil
from arm_env import ArmEnv
from RobotGame import RobotGame
from proportional import Experience

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 10000
MAX_EP_STEPS = 500
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 32
BATCH_SIZE = 16
VAR_MIN = 0.1
LOAD = True
SAVE = True
env = RobotGame()
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

#M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)
M = Experience(MEMORY_CAPACITY, BATCH_SIZE, 0.7)

saver = tf.train.Saver()
save_path = '/home/burak/catkin_ws/src/panda_sim/panda_sim_rl/weights/model_combinedReward_lessPenalty_randomReset'
load_path = '/home/burak/catkin_ws/src/panda_sim/panda_sim_rl/weights/model_combinedReward_lessPenalty_randomReset'

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(load_path))
else:
    sess.run(tf.global_variables_initializer())


def train():
    var = 20.  # control exploration
    mPointer = 0

    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        for t in range(MAX_EP_STEPS):

            # Added exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)
            #M.store_transition(s, a, r, s_)
            maxVal = M.get_max()
            M.add((s, a, r, s_), maxVal)
            mPointer += 1

            #if M.pointer > MEMORY_CAPACITY:
            if mPointer > MEMORY_CAPACITY:
                var = max([var*.9999, VAR_MIN])    # decay the action randomness
                # b_M = M.sample(BATCH_SIZE)
                # b_s = b_M[:, :STATE_DIM]
                # b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                # b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                # b_s_ = b_M[:, -STATE_DIM:]

                b_M, _, indices = M.select(0) 
                lul = M.get_max_data()
                #print lul[2]
                while b_M is None:
                    print "PRIORITY SICIS"
                    b_M, _, indices = M.select(0)
                b_s = np.zeros((BATCH_SIZE, STATE_DIM))
                b_a = np.zeros((BATCH_SIZE, ACTION_DIM))
                b_r = np.zeros((BATCH_SIZE, 1))
                b_s_ = np.zeros((BATCH_SIZE, STATE_DIM))
                for i in range(BATCH_SIZE):
                    data = b_M[i]
                    b_s[i, :] = data[0]
                    b_a[i, :] = data[1]
                    b_r[i, :] = data[2]
                    b_s_[i, :] = data[3]
                q_target = sess.run(critic.target_q, feed_dict={S: b_s, critic.a: b_a, R: b_r, S_: b_s_})
                q_current = sess.run(critic.q, feed_dict={S: b_s, critic.a: b_a, R: b_r, S_: b_s_})
                M.priority_update(indices, np.squeeze(np.abs(q_current-q_target)).tolist())

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS-1 or done:
                result = '| ----' 
                if info:
                    result = '| success (gently touched)' 
                elif done:
                    result = '| fail (object moved)'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      )
                break

        if SAVE and ep%50 == 0:
            save_model(save_path)
    if SAVE:
        save_model(save_path)

def save_model(lulpath):
    if os.path.isdir(lulpath): shutil.rmtree(lulpath)
    os.mkdir(lulpath)
    ckpt_path = os.path.join(lulpath, 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)

if __name__ == '__main__':
    # if LOAD:
    #     eval()
    # else:
    train()