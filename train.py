# -*- coding: utf-8 -*-
'''
June 2020 by Soohan Kang.
tngksapf@naver.com
https://github.com/
'''


from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

import numpy as np


#from hyperparams import Hyperparams as hp
from data_load import get_batch_data
from modules import *
import os, codecs
from tqdm import tqdm


class Model():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        self.global_step = 0
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data()

            else:
                #self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                #self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.x = tf.placeholder(tf.int32, shape=(None, 35))
                self.y = tf.placeholder(tf.int32, shape=(None, 35))

            with tf.variable_scope("embedding"):
                # Embedding
                #self.enc = embedding(self.x, 
                #                      vocab_size=hp.vocab, 
                #                      num_units=hp.hidden_units, 
                #                      scope="enc_embed")

                self.enc = embedding(self.x, 
                                      vocab_size=10000, 
                                      num_units=1024, 
                                      scope="enc_embed")

            config=None

            with tf.variable_scope("LSTM"):
                output, state = self._build_rnn_graph(self.enc, config, is_training)

            with tf.variable_scope("FC"):
                output = tf.contrib.layers.fully_connected(output, num_outputs = 10000)

                self.loss = tf.contrib.seq2seq.sequence_loss(logits = output, 
                       targets = self.y,
                       weights = tf.ones([1024, 35]))

                self.loss = tf.reduce_sum(self.loss)



            self.learning_rate = tf.constant(1.0)
            self.learning_rate = self._get_learning_rate_warmup()

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
               
            # Summary 
            tf.summary.scalar('loss', self.loss)



    def _build_rnn_graph(self, inputs, config, is_training):
        return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def make_cell():
            cell = tf.contrib.rnn.IndyLSTMCell(
                1024, forget_bias=1.0, reuse=not is_training)
            if is_training:
            #if is_training and config.keep_prob < 1:
                #cell = tf.contrib.rnn.DropoutWrapper(
                #    cell, output_keep_prob=config.keep_prob)

                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=0.8)
            return cell

#        cell = tf.contrib.rnn.MultiRNNCell(
#            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(2)], state_is_tuple=True)

        output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

        return output, state

    def _get_learning_rate_warmup(self):
        """Get learning rate warmup."""
        #warmup_steps = hparams.warmup_steps
        warmup_steps = 4000

        warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
        inv_decay = warmup_factor**(
            tf.to_float(warmup_steps - self.global_step))

        self.global_step += 200

        print(self.global_step)

        return tf.cond(
            tf.constant(self.global_step) < tf.constant(warmup_steps),
            lambda: inv_decay * self.learning_rate,
            lambda: self.learning_rate,
            name="learning_rate_warmup_cond")






if __name__ == "__main__":
    g = Model("train"); print("Graph loaded")

    sv = tf.train.Supervisor(graph=g.graph,
#                            logdir = hp.logdir)
                            logdir = "./test",
                            save_model_secs=0)

    with sv.managed_session() as sess:
        for epoch in range(1, 25):
            cost = 0
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g._get_learning_rate_warmup)
                sess.run(g.train_op)
                cost += sess.run(g.loss)


                tf.summary.scalar('loss', cost)
                g.merged = tf.summary.merge_all()

            
            print(cost)

            gs = sess.run(g.global_step)
            sv.saver.save(sess, './test/model_epoch_%02d_gs_%s' %(epoch, gs))



    print("Done")




