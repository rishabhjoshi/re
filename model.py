import tensorflow as tf
import numpy as np
import datetime
import time
import os

import attention
from settings import Setting

class GRU:
    def __init__(self, is_training, word_embeddings, word_pos1, word_pos2, settings):

        self.num_steps = num_steps = settings.num_steps
        self.voc_size = voc_size = settings.voc_size
        self.num_classes = num_classes = settings.num_classes
        self.gru_size = gru_size = settings.gru_size
        self.batch_size = batch_size = settings.batch_size

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_word')
        #position~
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos2')
        #~position
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        #self.total_shape = tf.placeholder(dtype=tf.int32, shape=[batch_size+1], name='total_shape')
        
        # pay more attention to this value!
        # since multi-instance multi-label, according to the previous data structure design
        # in each batch, the real number of sentences is total_num, not 'batch_num'(batch_size)
        total_num = settings.batch_size

        word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embeddings')
        #pos1_embedding = tf.Variable(tf.random_uniform([settings.pos_num, settings.pos_size],-1.0,1.0))
        #pos2_embedding = tf.Variable(tf.random_uniform([settings.pos_num, settings.pos_size],-1.0,1.0))
        pos1_embedding = tf.get_variable(initializer=word_pos1, name='word_pos1')
        pos2_embedding = tf.get_variable(initializer=word_pos2, name='word_pos2')

        #tf.contrib.rnn.BasicLSTMCell(...), there are also some others cells.
        gru_cell_forward = tf.contrib.rnn.GRUCell(gru_size)
        gru_cell_backward = tf.contrib.rnn.GRUCell(gru_size)

        #during the training process, using DROPOUT strategy. If not, skip.
        if is_training and settings.keep_prob < 1:
            gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=settings.keep_prob)
            gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=settings.keep_prob)

        cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward]*settings.num_layer)
        cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward]*settings.num_layer)

        self._initial_state_forward = cell_forward.zero_state(total_num,tf.float32)
        self._initial_state_backward = cell_backward.zero_state(total_num,tf.float32)

        #embedding layer bidirectional
        # [word_embeddding, pos1_embedding, pos2_embedding]
        input_forward = tf.concat([tf.nn.embedding_lookup(word_embedding, self.input_word), \
                                   tf.nn.embedding_lookup(pos1_embedding, self.input_pos1), \
                                   tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)],2)
        input_backward = tf.concat([tf.nn.embedding_lookup(word_embeddings, tf.reverse(self.input_word,[1])),\
                                    tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos1,[1])),\
                                    tf.nn.embedding_lookup(pos2_embedding,tf.reverse(self.input_pos2, [1]))], 2)

        output_forward = []
        output_backward = []
        state_forward = self._initial_state_forward
        state_backward = self._initial_state_backward

        with tf.variable_scope('GRU_FORWARD'):
            for step in range(num_steps):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                # step here is the length of each training data.(the length of sentence, number of words)
                # shape = [total_num, step, hidden_size]
                (cell_output_forward, state_forward) = cell_forward(input_forward[:, step, :], state_forward)
                output_forward.append(cell_output_forward)

        with tf.variable_scope('GRU_BACKWARD'):
            for step in range(num_steps):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output_backward, state_backward) = cell_backward(input_backward[:, step, :], state_backward)
                #[num_steps][total_num,gru_size]
                output_backward.append(cell_output_backward)
        
        #[num_steps][total_num,gru_size]---(concat)---> [num_steps][total_num*gru_size]
        #---reshape--->total_num*num_steps, gru_size
        output_forward = tf.reshape(tf.concat(output_forward, 1), [total_num, num_steps, gru_size])
        output_backward = tf.reshape(tf.concat(output_backward, 1), [total_num, num_steps, gru_size])
       
        #output_h = tf.add(output_forward,output_backward)  #wrong, we need to concat them, not add them!
        output_h = tf.concat([output_forward, output_backward], 2)

        #attention_layer return:[total_num, 1]
        output_att = attention.Attention_layer(output_h)
        #attention_layer
        #output_att = attention.Syn_attention_layer(output_h, self.sdp)
        
        output_drop = tf.nn.dropout(output_att, settings.keep_prob)

        W = tf.Variable(tf.truncated_normal([output_drop.get_shape()[1].value, num_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        y_hat = tf.nn.xw_plus_b(output_drop, W, b)
      
        output = []
        for i in range(self.batch_size):
            output.append(tf.nn.softmax(y_hat[i]))
        #output = y_hat

        self.loss = 0.0
        self.accuracy = 0.0

        self.probility = output
        
        # batch_avg_loss
        loss_origin = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.input_y))
        l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),weights_list=tf.trainable_variables())
        self.loss = loss_origin + l2_loss

        tf.summary.scalar('loss', self.loss)
        self.comparison = tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(self.input_y, 1)), tf.float32)
        #self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(self.input_y, 1)), tf.float32))
        self.accuracy = tf.reduce_mean(self.comparison)
        self.difference =  tf.reduce_sum(tf.squared_difference(output, self.input_y), 1)

        



