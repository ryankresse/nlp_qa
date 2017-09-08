from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import pdb
from evaluate import exact_match_score, f1_score, get_f1_score, sigmoid
from util import Progbar, minibatches, softmax

logger = logging.getLogger("hw3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn




class QASystem(object):
    def __init__(self, *args):

        self.FLAGS = args[0]
        embed_path = args[1]
        self.idx_word = args[2]
        self.val_only = args[3]
        self.tr_set_size = args[4]
        self.is_test = args[5]

        self.pretrained_embeddings = np.load(embed_path, mmap_mode='r')['glove'].astype(np.float32)
        # ==== set up placeholder tokens ========

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.contrib.layers.xavier_initializer()):
            self.add_weights()
            self.add_biases()
            self.setup_system()

    def add_weights(self):
        with tf.variable_scope('weights') as scope:
            self.weights = {
                'beg_mlp_weight1': tf.get_variable('beg_mlp_weight1',shape=[self.FLAGS.state_size*2, self.FLAGS.state_size*2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'end_mlp_weight1': tf.get_variable('end_mlp_weight1',shape=[self.FLAGS.state_size*2, self.FLAGS.state_size*2], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()),
                'beg_mlp_weight2': tf.get_variable('beg_mlp_weight2',shape=[self.FLAGS.state_size*2, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'end_mlp_weight2': tf.get_variable('end_mlp_weight2',shape=[self.FLAGS.state_size*2, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'full_att_weight': tf.get_variable('full_att_weight', shape=[self.FLAGS.num_perspectives, self.FLAGS.state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'max_att_weight': tf.get_variable('max_att_weight', shape=[self.FLAGS.num_perspectives, self.FLAGS.state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                'mean_att_weight': tf.get_variable('mean_att_weight', shape=[self.FLAGS.num_perspectives, self.FLAGS.state_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                }

    def add_biases(self):
        with tf.variable_scope('biases') as scope:
            self.biases = {
                'beg_mlp_bias1': tf.get_variable('beg_mlp_bias1', shape=[self.FLAGS.cont_length, 1], dtype=tf.float32),
                'beg_mlp_bias2': tf.get_variable('beg_mlp_bias2', shape=[self.FLAGS.cont_length, 1], dtype=tf.float32),
                'end_mlp_bias1': tf.get_variable('end_mlp_bias1', shape=[self.FLAGS.cont_length, 1], dtype=tf.float32),
                'end_mlp_bias2': tf.get_variable('end_mlp_bias2', shape=[self.FLAGS.cont_length, 1], dtype=tf.float32)
                }

    def create_datasets(self):

        self.quest_data_placeholder =  tf.placeholder(tf.int32, shape=(None, self.FLAGS.quest_length))
        self.cont_data_placeholder = tf.placeholder(tf.int32, shape=(None, self.FLAGS.cont_length))
        self.ans_data_placeholder = tf.placeholder(tf.int32, shape=(None, 2))

        if not self.is_test:
            shapes = (tf.TensorShape([None, self.FLAGS.quest_length]), tf.TensorShape([None, self.FLAGS.cont_length]), tf.TensorShape([None, self.FLAGS.ans_length]))
            self.iterator = tf.contrib.data.Iterator.from_structure((tf.int32, tf.int32, tf.int32), shapes)
        else:
            shapes = (tf.TensorShape([None, self.FLAGS.quest_length]), tf.TensorShape([None, self.FLAGS.cont_length]))
            self.iterator = tf.contrib.data.Iterator.from_structure((tf.int32, tf.int32), shapes)


        if not self.val_only and not self.is_test:
            self.tr_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.quest_data_placeholder, self.cont_data_placeholder, self.ans_data_placeholder))
            self.tr_dataset = self.tr_dataset.shuffle(buffer_size=self.tr_set_size)
            self.tr_dataset = self.tr_dataset.batch(self.FLAGS.batch_size)
            self.tr_inititializer = self.iterator.make_initializer(self.tr_dataset)

        if not self.is_test:
            self.val_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.quest_data_placeholder, self.cont_data_placeholder, self.ans_data_placeholder))
            self.val_dataset = self.val_dataset.batch(self.FLAGS.batch_size)
            self.val_inititializer = self.iterator.make_initializer(self.val_dataset)
            self.val_iterator =self.val_dataset.make_initializable_iterator()

        if self.is_test:
            self.test_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.quest_data_placeholder, self.cont_data_placeholder))
            self.test_dataset = self.test_dataset.batch(self.FLAGS.batch_size)
            self.test_inititializer = self.iterator.make_initializer(self.test_dataset)
            self.test_iterator =self.test_dataset.make_initializable_iterator()


    def setup_system(self):

        self.create_datasets()
        self.lr = self.FLAGS.learning_rate

        self.beg_logits, self.end_logits = self.add_prediction_op()

        self.beg_prob = tf.nn.softmax(self.beg_logits)
        self.end_prob = tf.nn.softmax(self.end_logits)
        self.starts = self.get_pred(self.beg_prob)
        self.ends = self.get_end_pred(self.end_prob, self.starts)

        if not self.is_test:
            self.loss = self.get_loss(self.beg_logits, self.end_logits)
            tf.summary.scalar('loss', self.loss)
            #self.train_op, self.grad_norm = self.add_train_op(self.loss)
            self.train_op = self.add_train_op(self.loss)


        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.FLAGS.summaries_dir)
        self.saver = tf.train.Saver()

    def apply_mask(self, items):
        items_flat = tf.reshape(items, [-1])
        cont_flat = tf.reshape(self.cont, [-1])
        self.mask = tf.sign(tf.cast(cont_flat, dtype=tf.float32))
        masked_items = items_flat * mask
        masked_items = tf.reshape(masked_items, tf.shape(items))
        return masked_items, mask

    def clip_labels(self, labels):
        return tf.clip_by_value(labels, 0, self.FLAGS.cont_length-1)

    def get_loss(self, beg_logits, end_logits):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            self.beg_labels, self.end_labels = self.get_labels(self.ans)
            self.beg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.beg_labels, logits=beg_logits)
            self.end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.end_labels, logits=end_logits)
            return tf.reduce_mean(self.beg_loss + self.end_loss)


    def add_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embeddings = self.pretrained_embeddings
            quest_embed = tf.nn.embedding_lookup(embeddings, self.quest)
            cont_embed = tf.nn.embedding_lookup(embeddings, self.cont)
        return (quest_embed, cont_embed)

    def add_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        #gradients, var = zip(*optimizer.compute_gradients(loss))
        #self.clip_val = tf.constant(self.FLAGS.max_gradient_norm, tf.float32)
        #gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_val)
        #train_op = optimizer.apply_gradients(zip(gradients, var))
        train_op = optimizer.minimize(loss)
        return train_op

    def input_lstm(self, quest_embed, quest_lens, filtered_cont, cont_lens):
        with tf.variable_scope('input_lstm', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size)

            (quest_output_fw, quest_output_bw), _= tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, quest_embed, sequence_length=quest_lens, dtype=tf.float32)
            quest_last_fw = tf.slice(quest_output_fw, [0, self.FLAGS.cont_length-1, 0], [-1, 1, -1])
            quest_last_bw = tf.slice(quest_output_bw, [0, 0, 0], [-1, 1, -1])
            quest_last_hid = tf.concat([quest_last_fw, quest_last_bw], 2)

            (cont_output_fw, cont_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, filtered_cont, sequence_length=cont_lens, dtype=tf.float32)
            return quest_last_hid, quest_output_fw, quest_output_bw, cont_output_fw, cont_output_bw


    def aggregate(self, att_vecs):
        with tf.variable_scope('agg_lstm', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            att_vecs = tf.transpose(att_vecs, [0, 2, 1]) #(batch, cont, num_per*6)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size)
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, att_vecs, sequence_length=self.cont_lens, dtype=tf.float32)
            return tf.nn.dropout(tf.concat([outputs[0], outputs[1]], 2), 0.8) #(batch, cont, state*2)

    def beg_lstm(self, cont_scaled):
        with tf.variable_scope('beg_lstm') as scope:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size*2)
            beg_lstm_output, hidden_state = tf.nn.dynamic_rnn(cell, cont_scaled, dtype=tf.float32, sequence_length=self.cont_lens)
            return beg_lstm_output

    def end_lstm(self, cont_scaled):
        with tf.variable_scope('end_lstm') as scope:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size*2)
            end_lstm_output, hidden_state = tf.nn.dynamic_rnn(cell, cont_scaled, dtype=tf.float32, sequence_length=self.cont_lens)
            return end_lstm_output


    def get_logits(self, raw_output, weights, bias, scope_name):
        #raw_output is (batch, quest+cont_length, embed_size)
        with tf.variable_scope(scope_name) as scope:
            logits = []
            for example in np.arange(self.FLAGS.batch_size):
                ex = tf.squeeze(tf.slice(raw_output, [example, 0, 0], [1, -1,-1])) #(cont_length, state_size*2).dot(self.FLAGS.state_size*2, state*2) = (cont_length, state*2)
                hid = tf.nn.dropout(tf.nn.relu(tf.matmul(ex, weights[0]) + bias[0]), 0.8) # #(cont_length, state_size*2).dot(self.FLAGS.state_size*2, 1) = (cont_length, 1)
                out = tf.matmul(hid, weights[1]) + bias[1]  #(cont_length, state_size*2).dot(self.FLAGS.state_size*2, 1) = (cont_length, 1)

                logits.append(out) #(1, 300)
            return tf.squeeze(tf.stack(logits))

    def filter_cont(self, quest, cont):
        with tf.variable_scope('filter_cont') as scope:
            quest_norm = tf.nn.l2_normalize(quest, dim=2)
            cont_norm = tf.nn.l2_normalize(cont, dim=2)
            all_filtered = []
            for example in np.arange(self.FLAGS.batch_size):
                ex_cont_norm = tf.squeeze(tf.slice(cont_norm, [example, 0, 0], [1, -1,-1])) #(cont, state)
                ex_quest_norm = tf.squeeze(tf.slice(quest_norm, [example, 0, 0], [1, -1,-1])) #(quest, state)
                prod = tf.matmul(ex_cont_norm, tf.transpose(ex_quest_norm)) #(cont, quest)
                relev = tf.reduce_max(prod, axis=1) #(cont)
                ex_cont = tf.squeeze(tf.slice(cont_norm, [example, 0, 0], [1, -1,-1])) #(cont, state)
                all_filtered.append(tf.multiply(ex_cont, tf.expand_dims(relev, 1))) #(cont, state)
            return tf.stack(all_filtered)

    def calculate_att_vectors(self, quest_last_hid, quest_out_fw, quest_out_bw, cont_out_fw, cont_out_bw):
        to_return = []
        for example in np.arange(self.FLAGS.batch_size):
            cont_reps_fw = tf.squeeze(tf.slice(cont_out_fw, [example, 0, 0], [1,-1,-1])) #(cont_len, state)
            cont_reps_bw = tf.squeeze(tf.slice(cont_out_bw, [example, 0, 0], [1,-1,-1])) #(cont_len, state)
            quest_reps_fw = tf.squeeze(tf.slice(quest_out_fw, [example, 0, 0], [1,-1,-1])) #(quest_len, state)
            quest_reps_bw = tf.squeeze(tf.slice(quest_out_bw, [example, 0, 0], [1,-1,-1])) #(quest_len, state)

            full_atts = self.get_full_atts(quest_reps_fw, quest_reps_bw, cont_reps_fw, cont_reps_bw)
            max_atts = self.get_reduced_atts(quest_reps_fw, quest_reps_bw, cont_reps_fw, cont_reps_bw, 'max_atts', self.weights['max_att_weight'], tf.reduce_max)
            mean_atts = self.get_reduced_atts(quest_reps_fw, quest_reps_bw, cont_reps_fw, cont_reps_bw, 'mean_atts', self.weights['mean_att_weight'], tf.reduce_mean)

            to_return.append(tf.concat([full_atts, max_atts, mean_atts], axis=0))
        return tf.nn.dropout(tf.stack(to_return), 0.8)


    def get_full_atts(self, quest_reps_fw, quest_reps_bw, cont_reps_fw, cont_reps_bw):

        last_quest_reps_fw = tf.slice(quest_reps_fw, [self.FLAGS.quest_length-1, 0], [1,-1]) #(1, state)
        last_quest_reps_bw = tf.slice(quest_reps_fw, [0, 0], [1,-1]) #(1, state)
        full_att_fw = self.calc_full_atts(last_quest_reps_fw, cont_reps_fw, self.weights['full_att_weight'], 'fw')
        full_att_bw = self.calc_full_atts(last_quest_reps_bw, cont_reps_bw, self.weights['full_att_weight'], 'bw')
        return tf.concat([full_att_fw, full_att_bw], 0) #(num_per, cont)

    def calc_full_atts(self, last_quest_reps, cont_reps, weights, scope_direction):
        with tf.variable_scope('full_att_{}'.format(scope_direction)) as scope:
            last_quest_reps = tf.multiply(last_quest_reps, weights)
            cont_reps_expand = tf.expand_dims(cont_reps, axis=1)
            weights_expand = tf.expand_dims(weights, axis=0)
            cont_reps_scaled = tf.multiply(cont_reps_expand, weights_expand) # (cont, 1, state) * (1, num_per, state) = (cont, per, state) For each cont_rep, we have N perspectives, each of which is state length long
            cont_reps_scaled = tf.nn.l2_normalize(cont_reps_scaled, dim=2) #
            last_quest_reps = tf.nn.l2_normalize(last_quest_reps, dim=1) #
            last_quest_reps_expand = tf.expand_dims(last_quest_reps, dim=0) # (1, num_per, state)
            prod  = tf.multiply(cont_reps_scaled, last_quest_reps_expand) # (cont, per, state) * (1, num_per, state) =  (cont, num_per, state) For cont, for each per, multiply it by the corresponding quest_rep perspective
            summed = tf.reduce_sum(prod, axis=2) # reduce along the state axis to finish the dot product = (cont, num_per)
            return tf.transpose(summed)

    def get_reduced_atts(self, quest_reps_fw, quest_reps_bw, cont_reps_fw, cont_reps_bw, scope_name, weights, reduce_fn):
        with tf.variable_scope(scope_name) as scope:
            fw_atts = self.calc_reduced_atts(quest_reps_fw, cont_reps_fw, 'fw', weights, reduce_fn)
            bw_atts = self.calc_reduced_atts(quest_reps_bw, cont_reps_bw, 'bw', weights, reduce_fn)
            return tf.concat([fw_atts, bw_atts], 0)

    def calc_reduced_atts(self, quest_reps, cont_reps, scope_dir, weights, reduce_fn):
        with tf.variable_scope(scope_dir) as scope:
            cont_reps_expand = tf.expand_dims(cont_reps, axis=1)
            weights_expand = tf.expand_dims(weights, axis=0)
            cont_reps_scaled = tf.multiply(cont_reps_expand, weights_expand) # (cont, 1, state) * (1, num_per, state) = (cont, per, state) For each cont_rep, we have N perspectives, each of which is state length long
            cont_reps_scaled = tf.nn.l2_normalize(cont_reps_scaled, dim=2) #
            quest_reps_expand = tf.expand_dims(quest_reps, axis=1) #
            quest_reps_scaled = tf.multiply(quest_reps_expand, weights_expand) # (cont, 1, state) * (1, num_per, state) = (quest, per, state) For each cont_rep, we have N perspectives, each of which is state length long
            quest_reps_scaled = tf.nn.l2_normalize(quest_reps_scaled, dim=2) #

            #you want to multiply each cont by every quest. (cont, num_per, state) * (quest, num_per, state)
            all_reduced = []
            for i in np.arange(self.FLAGS.num_perspectives):
                quest_per = tf.squeeze(tf.slice(quest_reps_scaled, [0, i, 0], [-1,1,-1])) # (quest, state)
                cont_per = tf.squeeze(tf.slice(cont_reps_scaled, [0, i, 0], [-1,1,-1])) # (cont, state)
                prod = tf.matmul(cont_per, tf.transpose(quest_per)) # (cont, state).dot(state, quest) = (cont, quest)
                all_reduced.append(reduce_fn(prod, axis=1)) # (cont, 1s)

            return tf.stack(all_reduced) # (num_per, cont)

    def scale_cont(self, cont, att):
        scaled = []
        for example in np.arange(self.FLAGS.batch_size):
            ex = tf.slice(cont,[example, 0, 0], [1, -1, -1])
            ex = tf.squeeze(ex)
            att_vec = tf.slice(att, [example,0], [1, -1])
            att_mat = tf.tile(att_vec, [self.FLAGS.state_size*2, 1])
            scaled.append(tf.multiply(ex, tf.transpose(att_mat)))
        return tf.stack(scaled)

    def get_lens(self):
        quest_mask = tf.sign(self.quest)
        cont_mask = tf.sign(self.cont)
        quest_lens = tf.reduce_sum(quest_mask,axis=1)
        cont_lens = tf.reduce_sum(cont_mask ,axis=1)
        return quest_lens, cont_lens

    def get_labels(self, labels):
        labels = self.clip_labels(labels)
        beg_labels = tf.squeeze(tf.slice(labels, [0, 0], [-1,1]))
        end_labels = tf.squeeze(tf.slice(labels, [0, 1], [-1,1]))
        return beg_labels, end_labels

    def add_prediction_op(self):

        if not self.is_test:
            self.quest, self.cont, self.ans = self.iterator.get_next()
        else:
            self.quest, self.cont = self.iterator.get_next()

        self.quest_lens, self.cont_lens = self.get_lens()

        self.quest_embed, self.cont_embed = self.add_embeddings()
        self.filtered_cont = self.filter_cont(self.quest_embed, self.cont_embed)

        self.quest_last_hid, self.quest_out_fw, self.quest_out_bw, self.cont_out_fw, self.cont_out_bw = self.input_lstm(self.quest_embed, self.quest_lens,  self.filtered_cont, self.cont_lens)

        self.att_vectors = self.calculate_att_vectors(self.quest_last_hid, self.quest_out_fw, self.quest_out_bw, self.cont_out_fw, self.cont_out_bw) # (batch, num_per*6, cont)
        self.aggregated = self.aggregate(self.att_vectors)

        self.beg_logits = self.get_logits(self.aggregated, [self.weights['beg_mlp_weight1'], self.weights['beg_mlp_weight2']], [self.biases['beg_mlp_bias1'], self.biases['beg_mlp_bias2']], 'beg_logits') #(state*2)
        self.end_logits = self.get_logits(self.aggregated,[self.weights['end_mlp_weight1'], self.weights['end_mlp_weight2']], [self.biases['end_mlp_bias1'], self.biases['end_mlp_bias2']], 'end_logits')
        return self.beg_logits, self.end_logits

    def get_pred(self, probs):
        return tf.argmax(probs, axis=1)

    def get_end_pred(self, probs, start_pos):
        self.ranges = tf.cast(tf.tile(tf.expand_dims(tf.range(self.FLAGS.cont_length), 0), [self.FLAGS.batch_size,1]), tf.int64)
        self.pred_mask = tf.less(self.ranges, tf.expand_dims(start_pos, 1))
        self.masked_probs = tf.where(self.pred_mask, tf.zeros((self.FLAGS.batch_size, self.FLAGS.cont_length)), probs)
        return tf.argmax(self.masked_probs, axis=1)

    def evaluate_performance(self, pred_ans_words, ans_text):
        f1 =  get_f1_score(pred_ans_words, ans_text.tolist())
        return f1

    def write_summaries(self, summaries, epoch, batch, num_batches):
        self.summary_writer.add_summary(summaries, (epoch * num_batches) + batch)

    def add_weights_bias_summary(self):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

    def variable_summaries(self, name, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            tf.summary.histogram('histogram', var)

    def get_ans_words(self, starts_ends, cont):
        words = []

        for ix in np.arange(starts_ends.shape[0]):
            start = starts_ends[ix, 0]; end = starts_ends[ix,1];
            if start > end:
                words.append([])
            elif start == end:
                #select the token from the training example at the start position, then get the word for it
                words.append([self.idx_word[cont[ix, start]]])
            else:
                tokens = cont[ix, start:end + 1]
                words.append([self.idx_word[tok] for tok in tokens])

        return [" ".join(w) for w in words]

    def get_f1(self, cont, starts, ends, ans_text):
        pred_words = self.get_ans_words(np.hstack([np.expand_dims(starts,1), np.expand_dims(ends,1)]), cont)
        f1 = self.evaluate_performance(pred_words, ans_text)
        return f1

    def compute_and_report_epoch_stats(self, epoch, running_loss, running_f1, num_batches, report_type='train'):
        avg_loss = running_loss / num_batches
        print('average {} loss for epoch {}: {:.2E}'.format(report_type, epoch, running_loss / num_batches))
        avg_f1 = running_f1 / num_batches
        print('average {} f1 for epoch {}: {}'.format(report_type, epoch, avg_f1))
        print("")
        return avg_loss, avg_f1



    def train_on_batch(self, sess):
        #to_run = [self.train_op, self.ans, self.loss, self.beg_logits, self.end_logits, self.beg_prob, self.end_prob, self.grad_norm, self.merged]
        to_run = [self.train_op, self.ans, self.loss, self.beg_logits, self.end_logits, self.beg_prob, self.end_prob, self.merged]

        #train_op, ans, loss, beg_logits, end_logits, beg_prob, end_prob, grad_norm, merged =  sess.run(to_run)
        #return loss, beg_logits, end_logits, beg_prob, end_prob, grad_norm, merged
        train_op, ans, loss, beg_logits, end_logits, beg_prob, end_prob, merged =  sess.run(to_run)
        return loss, beg_logits, end_logits, beg_prob, end_prob, merged


    def validate_on_batch(self, sess):
        loss, ans, cont, starts, ends  =  sess.run([self.loss, self.ans, self.cont, self.starts, self.ends])
        return loss, ans, cont, starts, ends

    def validate(self, sess, val_set, epoch):
        num_batches = int(len(val_set[0]) / self.FLAGS.batch_size)
        running_loss = 0
        feed_dict= {
            self.quest_data_placeholder: val_set[0],
            self.cont_data_placeholder:val_set[1],
            self.ans_data_placeholder:val_set[2]
        }
        sess.run(self.val_inititializer, feed_dict=feed_dict)
        all_starts = []
        all_ends = []
        for i in range(num_batches -1):
            print('Batch {} of {}'.format(i+1, num_batches))
            if (i == num_batches-1): break
            loss, ans, cont, starts, ends = self.validate_on_batch(sess)
            all_starts.append(starts)
            all_ends.append(ends)
            running_loss +=loss

        all_starts = np.hstack(all_starts)
        all_ends = np.hstack(all_ends)
        f1 = self.get_f1(val_set[1], all_starts, all_ends, val_set[4])
        avg_loss = running_loss / num_batches
        print('Epoch {} val loss: {:.2E}, f1: {}'.format(epoch, avg_loss, f1))
        print('=========================')
        return avg_loss, f1

    def test_on_batch(self, sess):
        starts, ends  =  sess.run([self.starts, self.ends])
        return starts, ends

    def test(self, sess, test_set):
        num_batches = int(len(test_set[0]) / self.FLAGS.batch_size)
        feed_dict= {
            self.quest_data_placeholder: test_set[1],
            self.cont_data_placeholder:test_set[0],
        }
        sess.run(self.test_inititializer, feed_dict=feed_dict)
        all_starts = []
        all_ends = []
        for i in range(num_batches):
            print('Batch {} of {}'.format(i+1, num_batches))
            starts, ends = self.test_on_batch(sess)
            all_starts.append(starts)
            all_ends.append(ends)
        all_starts = np.hstack(all_starts)
        all_ends = np.hstack(all_ends)
        return all_starts, all_ends


    def run_epoch(self, sess, train_examples, epoch):
        num_batches = int(len(train_examples[0]) / self.FLAGS.batch_size)
        running_loss = 0; running_f1 = 0;
        for i in range(num_batches - 1):
            print('Batch {} of {}'.format(i+1, num_batches))
            sess.run(self.tr_inititializer, feed_dict={self.quest_data_placeholder: train_examples[0], self.cont_data_placeholder:train_examples[1], self.ans_data_placeholder:train_examples[2]})
            if (i == num_batches - 1): break
            #loss, beg_logits, end_logits, beg_prob, end_prob, grad_norm, merged  = self.train_on_batch(sess)
            loss, beg_logits, end_logits, beg_prob, end_prob, merged  = self.train_on_batch(sess)

            running_loss +=loss
            #print('loss: {:.2E}, grad_norm: {}'.format(loss, grad_norm))
            print('loss: {:.2E}'.format(loss))

            if i % 200 == 0:
                self.write_summaries(merged, epoch, i, num_batches)

        avg_loss = running_loss / num_batches

        print('Epoch {} train loss: {:.2E}'.format(epoch, avg_loss))
        print('=========================')
        return avg_loss, beg_prob, end_prob

        #return avg_loss, grad_norm, beg_prob, end_prob


    def retrieve_prev_best_score(self):
        with open(self.FLAGS.prev_best_score_file, 'r') as the_file:
            score_epoch = the_file.readline()
        if score_epoch == '':
            best_score = np.inf
            epoch = 0
            print('no previous best score found. setting best score to {}'.format(best_score))
        else:
            arr = score_epoch.split(',')
            best_score = float(arr[0])
            epoch = int(arr[1])
            print('restored previous best score of {}'.format(best_score))
        print('=========')
        print('')
        return best_score, epoch

    def write_to_train_logs(self, tr_loss, val_loss, val_f1, epoch_dur):
        with open(self.FLAGS.train_stats_file, 'a') as f:
            f.write("{},{},{},{}\n".format(tr_loss, val_loss, val_f1, epoch_dur))

    def save_model(self, best_score, epoch, saver, sess):
        if saver:
            logger.info("New best score! Saving model.")
            saver.save(sess, self.FLAGS.train_dir+'/' + self.FLAGS.ckpt_file_name)
            with open('best_score.txt', 'w') as the_file:
                the_file.write(str(best_score) +',{}'.format(epoch))
        print('==========')
        print('')

    def write_val_summaries(self, sess, epoch, val_loss, val_f1):
        val_loss_tensor = tf.constant(val_loss)
        val_f1_tensor = tf.constant(val_f1)
        val_loss_summ = tf.summary.scalar('val_loss',val_loss_tensor)
        val_f1_summ = tf.summary.scalar('val_f1', val_f1_tensor)
        val_merged = tf.summary.merge([val_loss_summ, val_f1_summ])
        merged_for_write = sess.run(val_merged)
        self.summary_writer.add_summary(merged_for_write, epoch)

    def maybe_change_lr(self, best_score, num_since_improve):
        if num_since_improve > self.FLAGS.num_epochs_per_anneal:
            new_lr = self.lr / 10.0
            print("model hasn't proved on best score of {} in {} epochs. Annealing lr to {}".format(best_score, self.FLAGS.num_epochs_per_anneal,new_lr ))
            return new_lr, 0
        else:
            print('model hasn"t improved on best score of {} in {} epochs. Current lr is {}'.format(best_score, num_since_improve+1, self.lr))
            return self.lr, num_since_improve+1


    def fit(self, sess, saver, tr_set, val_set, train_dir):
        best_score, epoch_num = self.retrieve_prev_best_score()
        num_since_improve = 0

        for epoch in range(epoch_num, epoch_num + self.FLAGS.epochs):
            epoch_num += 1
            print('')
            logger.info("Training for epoch %d out of %d", epoch_num, epoch_num + self.FLAGS.epochs)
            print('==========')
            tic = time.time()

            #tr_loss, grad_norm, beg_prob, end_prob = self.run_epoch(sess, tr_set, epoch)
            tr_loss, beg_prob, end_prob = self.run_epoch(sess, tr_set, epoch)
            print('=========================')
            toc = time.time()
            epoch_dur = (toc - tic) / 60
            logger.info("Epoch training took {} minutes".format(epoch_dur))

            logger.info("Validating for epoch %d out of %d", epoch_num, epoch_num + self.FLAGS.epochs)
            print('==========')
            print('')

            val_loss, val_f1 = self.validate(sess, val_set, epoch)
            toc = time.time()
            epoch_dur = (toc - tic) / 60
            self.write_to_train_logs(tr_loss, val_loss, val_f1, epoch_dur)
            logger.info("Epoch took {} minutes".format(epoch_dur))

            if val_loss < best_score:
                best_score = val_loss
                num_since_improve = 0
                self.save_model(best_score, epoch_num, saver, sess)
            else:
                num_since_improve +=1
                if num_since_improve == 3:
                    print('Model has not improved in {} epochs. Stopping now. Bye bye.'.format(num_since_improve))
                    break
                self.lr *= .10
                print('Validation loss did not improve. Decreasing learning rate to {:.2E}'.format(self.lr))

        return best_score


    def train(self, session, tr_set, val_set, train_dir):
        saver = self.saver
        self.fit(session, saver, tr_set, val_set, train_dir)
