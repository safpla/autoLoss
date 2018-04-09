""" Decide which loss to update by Reinforce Learning """
# __Author__ == "Haowen Xu"
# __Data__ == "04-07-2018"

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Controller():
    def __init__(self, config, graph):
        self.config = config
        self.graph = graph
        self._build_placeholder()
        self._inference_graph()

    def _build_placeholder(self):
        config = self.config
        s_size = config.dim_state_rl
        with self.graph.as_default():
            self.state_plh = tf.placeholder(shape=[None, s_size],
                                        dtype=tf.float32)
            self.reward_plh = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_plh = tf.placeholder(shape=[None], dtype=tf.int32)

    def _inference_graph(self):
        config = self.config
        h_size = config.dim_hidden_rl
        a_size = config.dim_action_rl
        lr = config.lr_rl
        with self.graph.as_default():
            hidden = slim.fully_connected(self.state_plh, h_size,
                                        biases_initializer=None,
                                        activation_fn=tf.nn.relu)
            self.output = slim.fully_connected(hidden, a_size,
                                            biases_initializer=None,
                                            activation_fn=tf.nn.softmax)
            self.chosen_action = tf.argmax(self.output, 1)

            self.indexes = tf.range(0, tf.shape(self.output)[0])\
                * tf.shape(self.output)[1] + self.action_plh
            self.responsible_output = tf.gather(tf.reshape(self.output, [-1]),
                                                self.indexes)
            self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)
                                        * self.reward_plh)

            # restore gradients and update them after several iterals
            self.tvars = tf.trainable_variables()
            tvars = self.tvars
            self.gradient_plhs = []
            for idx, var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_plh')
                self.gradient_plhs.append(placeholder)

            self.gradients = tf.gradients(loss, tvars)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = optimizer.apply_gradients(zip(self.gradient_plhs, tvars))
            self.init = tf.global_variables_initializer()

    def get_gradients(self, sess, state, action, reward):
        """ Return the gradients according to one episode

        Args:
            sess: Current tf.Session
            state: shape = [time_steps, dim_state_rl]
            action: shape = [time_steps, dim_action_rl]
            reward: shape = [time_steps]

        Returns:
            grads: Gradients of all trainable variables
        """
        assert sess.graph is self.graph
        feed_dict = {self.reward_plh: reward,
                     self.action_plh: action,
                     self.state_plh: state}
        grads = sess.run(self.gradients, feed_dict=feed_dict)
        return grads

    def sample(self, sess, state):
        #
        # Sample an action from a given state, probabilistically

        # Args:
        #     sess: Current tf.Session
        #     state: shape = [dim_state_rl]

        # Returns:
        #     action: shape = [dim_action_rl]
        #
        assert sess.graph is self.graph
        a_dist = sess.run(self.output, feed_dict={state.state_plh: [state]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        return a
