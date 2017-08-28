from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

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


#this is where you should take your input and transform it into your symbolic representation
class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        return

#this takes your symbolic representation and produces our ouput probabilities.
class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        return

class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.FLAGS = args[0]
        embed_path = args[1]
        self.idx_word = args[2]
        self.tr_set = args[3]
        self.val_set = args[4]

        self.pretrained_embeddings = np.load(embed_path, mmap_mode='r')['glove']
        self.isTest = False
        # ==== set up placeholder tokens ========
        self.cont_placeholer = None
        self.quest_placeholder = None
        self.ans_placeholder = None
        self.dropout_placeholder = None


        #self.add_placeholders()
        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.contrib.layers.xavier_initializer()):
            self.add_weights()
            self.add_biases()
            self.setup_system()

            #self.setup_loss()
        # ==== set up training/updating procedure ====


    def add_weights(self):
        #out is (batch, quest_length, hidden_size*2)
        with tf.variable_scope('weights') as scope:
            self.weights = {
                'beg_mlp_weight1': tf.get_variable('beg_mlp_weight1',shape=[self.FLAGS.state_size*2, self.FLAGS.state_size*2], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()),
                'end_mlp_weight1': tf.get_variable('end_mlp_weight1',shape=[self.FLAGS.state_size*2, self.FLAGS.state_size*2], dtype=tf.float64,initializer=tf.contrib.layers.xavier_initializer()),
                'beg_mlp_weight2': tf.get_variable('beg_mlp_weight2',shape=[self.FLAGS.state_size*2, 1], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()),
                'end_mlp_weight2': tf.get_variable('end_mlp_weight2',shape=[self.FLAGS.state_size*2, 1], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()),
                'full_att_weight': tf.get_variable('full_att_weight', shape=[self.FLAGS.num_perspectives, self.FLAGS.state_size], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()),
                #'attention_weight': tf.get_variable('attention_weight', shape=[self.FLAGS.state_size*4, self.FLAGS.state_size*2], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()),
                'max_att_weight': tf.get_variable('max_att_weight', shape=[self.FLAGS.num_perspectives, self.FLAGS.state_size], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()),
                'mean_att_weight': tf.get_variable('mean_att_weight', shape=[self.FLAGS.num_perspectives, self.FLAGS.state_size], dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer()),
                }

    def add_biases(self):
        with tf.variable_scope('biases') as scope:
            self.biases = {
                'beg_mlp_bias1': tf.get_variable('beg_mlp_bias1', shape=[self.FLAGS.cont_length, 1], dtype=tf.float64),
                'beg_mlp_bias2': tf.get_variable('beg_mlp_bias2', shape=[self.FLAGS.cont_length, 1], dtype=tf.float64),
                'end_mlp_bias1': tf.get_variable('end_mlp_bias1', shape=[self.FLAGS.cont_length, 1], dtype=tf.float64),
                'end_mlp_bias2': tf.get_variable('end_mlp_bias2', shape=[self.FLAGS.cont_length, 1], dtype=tf.float64)
                }

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

          These placeholders are used as inputs by the rest of the model building and will be fed
          data during training.  Note that when "None" is in a placeholder's shape, it's flexible
          (so we can use different batch sizes without rebuilding the model).

          Adds following nodes to the computational graph
          self.quest_placeholder: (None, quest_length)
          self.context_placeholder: (None, context_length)
          self.ans_placeholder: (None, 2)
          dropout_placeholder: (scalar)
       """
        self.quest_placeholder = tf.placeholder(tf.int64, shape=(None, self.FLAGS.quest_length))
        self.cont_placeholder = tf.placeholder(tf.int64, shape=(None, self.FLAGS.cont_length))
        self.ans_placeholder = tf.placeholder(tf.int32, shape=(None, 2))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(None))


    def create_datasets(self):

        self.quest_data_placeholder =  tf.placeholder(tf.int64, shape=(None, self.FLAGS.quest_length))
        self.cont_data_placeholder = tf.placeholder(tf.int64, shape=(None, self.FLAGS.cont_length))
        self.ans_data_placeholder = tf.placeholder(tf.int32, shape=(None, 2))

        '''
        self.quest_text_placeholder =  tf.placeholder(tf.string, shape=(None))
        self.cont_text_placeholder = tf.placeholder(tf.string, shape=(None))
        self.ans_text_placeholder = tf.placeholder(tf.string, shape=(None))
        '''

        self.tr_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.quest_data_placeholder, self.cont_data_placeholder, self.ans_data_placeholder))
        self.tr_dataset = self.tr_dataset.shuffle(buffer_size=self.tr_set[0].shape[0])
        self.tr_dataset = self.tr_dataset.batch(self.FLAGS.batch_size)

        self.val_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.quest_data_placeholder, self.cont_data_placeholder, self.ans_data_placeholder))
        self.val_dataset = self.val_dataset.batch(self.FLAGS.batch_size)
        shapes = (tf.TensorShape([None, self.FLAGS.quest_length]), tf.TensorShape([None, self.FLAGS.cont_length]), tf.TensorShape([None, self.FLAGS.ans_length]))
        self.iterator = tf.contrib.data.Iterator.from_structure((tf.int64, tf.int64, tf.int32), shapes)
        self.tr_inititializer = self.iterator.make_initializer(self.tr_dataset)
        self.val_inititializer = self.iterator.make_initializer(self.val_dataset)

        #self.val_iterator =self.val_dataset.make_initializable_iterator()


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        '''
        self.tr_dataset = tf.contrib.data.Dataset.from_tensor_slices((self.quest_data_placeholder, self.cont_data_placeholder, self.ans_data_placeholder))
        self.tr_dataset = self.tr_dataset.shuffle(buffer_size=self.tr_set[0].shape[0])
        self.tr_dataset = self.tr_dataset.batch(self.FLAGS.batch_size)
        self.iterator =self.tr_dataset.make_initializable_iterator()
        '''
        self.create_datasets()
        self.lr = self.FLAGS.learning_rate
        self.beg_logits, self.end_logits = self.add_prediction_op()

        self.beg_labels, self.end_labels = self.get_labels(self.ans)
        self.loss = self.get_loss(self.beg_logits, self.end_logits, self.beg_labels, self.end_labels)
        tf.summary.scalar('loss', self.loss)
        self.train_op, self.grad_norm = self.add_train_op(self.loss)
        #tf.summary.scalar('grad_norm', self.grad_norm)

        self.beg_prob = tf.nn.softmax(self.beg_logits)
        self.end_prob = tf.nn.softmax(self.end_logits)
        #tf.summary.histogram('beg_prob', self.beg_prob)
        #tf.summary.histogram('end_prob', self.end_prob)

        self.starts = self.get_pred(self.end_prob)
        self.ends = self.get_pred(self.end_prob)

        #self.add_weights_bias_summary()

        #tf.summary.histogram('ans_len', self.ends - self.starts)
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.FLAGS.summaries_dir)
        self.saver = tf.train.Saver()

    def apply_mask(self, items):
        items_flat = tf.reshape(items, [-1])
        cont_flat = tf.reshape(self.cont, [-1])
        mask = tf.sign(tf.cast(cont_flat, dtype=tf.float64))
        masked_items = items_flat * mask
        masked_items = tf.reshape(masked_items, tf.shape(items))
        return masked_items

    def clip_labels(self, labels):
        return tf.clip_by_value(labels, 0, self.FLAGS.cont_length-1)


    def get_loss(self, beg_logits, end_logits, beg_labels, end_labels):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            #beg_logits = self.mask_logits(beg_logits)
            #end_logits = self.mask_logits(end_logits)
            self.beg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=beg_labels, logits=beg_logits)
            self.end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=end_labels, logits=end_logits)
            #example_sum_loss = tf.reduce_sum(beg_losses + end_losses, axis=1)
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
        gradients, var = zip(*optimizer.compute_gradients(loss))
        self.clip_val = tf.constant(self.FLAGS.max_gradient_norm, tf.float64) #tf.cond(loss > 50000, lambda: tf.constant(100.0, dtype=tf.float64), lambda: tf.constant(self.FLAGS.max_gradient_norm, dtype=tf.float64))

        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_val)

        train_op = optimizer.apply_gradients(zip(gradients, var))
        return train_op, grad_norm


    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs


    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em


    def create_feed_dict(self, quest_batch, cont_batch, ans_batch, dropout):
         """Creates the feed_dict for the dependency parser.

         A feed_dict takes the form of:

         feed_dict = {
                 <placeholder>: <tensor of values to be passed for placeholder>,
                 ....
         }

         Hint: The keys for the feed_dict should be a subset of the placeholder
                     tensors created in add_placeholders.
         Hint: When an argument is None, don't add it to the feed_dict.

         Args:
         Returns:
             feed_dict: The feed dictionary mapping from placeholders to values.
         """
         if ans_batch is None:
             feed_dict = {
                 self.quest_placeholder: quest_batch,
                 self.cont_placeholder: cont_batch,
                 self.dropout_placeholder: dropout
             }
         else:
             feed_dict = {
                 self.quest_placeholder: quest_batch,
                 self.cont_placeholder: cont_batch,
                 self.dropout_placeholder: dropout,
                 self.ans_placeholder: ans_batch
             }
         return feed_dict


    def input_lstm(self, quest_embed, quest_lens, filtered_cont, cont_lens):
        with tf.variable_scope('input_lstm', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size)

            (quest_output_fw, quest_output_bw), _= tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, quest_embed, sequence_length=quest_lens, dtype=tf.float64)
            quest_last_fw = tf.slice(quest_output_fw, [0, self.FLAGS.cont_length-1, 0], [-1, 1, -1])
            quest_last_bw = tf.slice(quest_output_bw, [0, 0, 0], [-1, 1, -1])
            quest_last_hid = tf.concat([quest_last_fw, quest_last_bw], 2)

            (cont_output_fw, cont_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, filtered_cont, sequence_length=cont_lens, dtype=tf.float64)
            return quest_last_hid, quest_output_fw, quest_output_bw, cont_output_fw, cont_output_bw


    def aggregate(self, att_vecs):
        with tf.variable_scope('agg_lstm', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            att_vecs = tf.transpose(att_vecs, [0, 2, 1]) #(batch, cont, num_per*6)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size)
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, att_vecs, sequence_length=self.cont_lens, dtype=tf.float64)
            return tf.concat([outputs[0], outputs[1]], 2) #(batch, cont, state*2)

    def beg_lstm(self, cont_scaled):
        with tf.variable_scope('beg_lstm') as scope:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size*2)
            beg_lstm_output, hidden_state = tf.nn.dynamic_rnn(cell, cont_scaled, dtype=tf.float64, sequence_length=self.cont_lens)
            return beg_lstm_output

    def end_lstm(self, cont_scaled):
        with tf.variable_scope('end_lstm') as scope:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size*2)
            end_lstm_output, hidden_state = tf.nn.dynamic_rnn(cell, cont_scaled, dtype=tf.float64, sequence_length=self.cont_lens)
            return end_lstm_output


    def get_logits(self, raw_output, weights, bias, scope_name):
        #raw_output is (batch, quest+cont_length, embed_size)
        with tf.variable_scope(scope_name) as scope:
            logits = []
            for example in np.arange(self.FLAGS.batch_size):
                ex = tf.squeeze(tf.slice(raw_output, [example, 0, 0], [1, -1,-1])) #(cont_length, state_size*2).dot(self.FLAGS.state_size*2, state*2) = (cont_length, state*2)
                hid = tf.nn.relu(tf.matmul(ex, weights[0]) + bias[0]) # #(cont_length, state_size*2).dot(self.FLAGS.state_size*2, 1) = (cont_length, 1)
                out = tf.matmul(hid, weights[1]) + bias[1]  #(cont_length, state_size*2).dot(self.FLAGS.state_size*2, 1) = (cont_length, 1)

                logits.append(out) #(1, 300)
            return tf.squeeze(tf.stack(logits))

    def get_beg_logits(self, raw_output):
        #raw_output is (batch, quest+cont_length, embed_size)
        with tf.variable_scope('beg_logits') as scope:
            logits = []
            for example in np.arange(self.FLAGS.batch_size):
                ex = tf.squeeze(tf.slice(raw_output, [example, 0, 0], [1, -1,-1])) #(cont_length, state_size*2)
                out = tf.nn.relu(tf.matmul(ex, self.weights['beg_mlp_weight1']) + self.biases['beg_mpl_bias1']) #(state*2)

                logits.append(out) #(1, 300)
            return tf.squeeze(tf.stack(logits))

    def get_end_logits(self, raw_output):
        with tf.variable_scope('end_logits') as scope:
            logits = []
            for example in np.arange(self.FLAGS.batch_size):
                ex = tf.squeeze(tf.slice(raw_output, [example, 0, 0], [1, -1,-1]))
                out = tf.nn.relu(tf.matmul(ex, self.weights['end_mlp_weight1']) + self.biases['end_mpl_bias1'])
                logits.append(out)
            return tf.squeeze(tf.stack(logits))


    def transform_scaled_cont_concat(self, cont_concat):
        with tf.variable_scope('reshape_scaled_cont_concat') as scope:
            reshaped = []
            for example in np.arange(self.FLAGS.batch_size):
                ex = tf.squeeze(tf.slice(cont_concat, [example, 0, 0], [1, -1,-1]))
                out = tf.nn.relu(tf.matmul(ex, self.weights['attention_weight'])) # (300, 800) * (800, 400)
                reshaped.append(out)
            return tf.squeeze(tf.stack(reshaped))

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
        return tf.stack(to_return)

    '''
    def cos_sim(self, first, second, scope_name):
        with tf.variable_scope(scope_name):
            first = tf.nn.l2_normalize(first, dim=1)
            second = tf.nn.l2_normalize(second, dim=1)
            cos_sim = tf.matmul(first, tf.transpose(second))
            return cos_sim
    '''

    def get_full_atts(self, quest_reps_fw, quest_reps_bw, cont_reps_fw, cont_reps_bw):

        last_quest_reps_fw = tf.slice(quest_reps_fw, [self.FLAGS.quest_length-1, 0], [1,-1]) #(1, state)
        last_quest_reps_bw = tf.slice(quest_reps_fw, [0, 0], [1,-1]) #(1, state)
        #last_quest_reps_fw = tf.multiply(last_quest_reps_fw, self.weights['full_att_weight']) # (1, state) * (num_per, state) = (num_per, state)

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
        #for example in batch

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
        beg_labels = tf.slice(labels, [0, 0], [-1,1])
        end_labels = tf.slice(labels, [0, 1], [-1,1])
        return tf.squeeze(beg_labels), tf.squeeze(end_labels)

    def add_prediction_op(self):


        self.quest, self.cont, self.ans = self.iterator.get_next()

        self.quest_lens, self.cont_lens = self.get_lens()

        self.quest_embed, self.cont_embed = self.add_embeddings()
        self.filtered_cont = self.filter_cont(self.quest_embed, self.cont_embed)

        self.quest_last_hid, self.quest_out_fw, self.quest_out_bw, self.cont_out_fw, self.cont_out_bw = self.input_lstm(self.quest_embed, self.quest_lens,  self.filtered_cont, self.cont_lens)

        self.att_vectors = self.calculate_att_vectors(self.quest_last_hid, self.quest_out_fw, self.quest_out_bw, self.cont_out_fw, self.cont_out_bw) # (batch, num_per*6, cont)
        self.aggregated = self.aggregate(self.att_vectors)

        self.beg_logits = self.apply_mask(self.get_logits(self.aggregated, [self.weights['beg_mlp_weight1'], self.weights['beg_mlp_weight2']], [self.biases['beg_mlp_bias1'], self.biases['beg_mlp_bias2']], 'beg_logits')) #(state*2)
        self.end_logits = self.apply_mask(self.get_logits(self.aggregated,[self.weights['end_mlp_weight1'], self.weights['end_mlp_weight2']], [self.biases['end_mlp_bias1'], self.biases['end_mlp_bias2']], 'end_logits'))
        return self.beg_logits, self.end_logits

    def debug_on_batch(self, sess, quest_batch, cont_batch, ans_batch):
        feed = self.create_feed_dict(quest_batch, cont_batch, ans_batch, self.FLAGS.dropout)
        quest_last_hid, cont_out = sess.run([self.self.quest_last_hid, self.cont_out], feed_dict=feed)
        _, loss, logits, grad_norm = sess.run([self.train_op, self.loss, self.pred, self.grad_norm], feed_dict=feed)
        return loss, logits, grad_norm

    def get_pred(self, probs):
        return tf.argmax(probs, axis=1)

    def evaluate_performance(self, pred_ans_words, ans_text):
        f1 =  get_f1_score(pred_ans_words, ans_text.tolist())
        return f1

    def write_summaries(self, summaries, epoch, batch, num_batches):
        self.summary_writer.add_summary(summaries, (epoch * num_batches) + batch)

    def add_weights_bias_summary(self):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        '''for k, v in self.weights.items():
            self.variable_summaries(k,v)
        for k, v in self.biases.items():
            self.variable_summaries(k,v)
        '''

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

    def write_prob(self, beg_prob, end_prob):
        np.save(self.FLAGS.beg_prob_file, beg_prob)
        np.save(self.FLAGS.end_prob_file, end_prob)

    def train_on_batch(self, sess):
        #logits = sess.run(self.beg_logits)
        #pdb.set_trace()
        to_run = [self.train_op, self.ans, self.loss, self.beg_logits, self.end_logits, self.beg_prob, self.end_prob, self.grad_norm, self.merged]
        train_op, ans, loss, beg_logits, end_logits, beg_prob, end_prob, grad_norm, merged =  sess.run(to_run)
        return loss, beg_logits, end_logits, beg_prob, end_prob, grad_norm, merged

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
            #if i == 2: break
            if (i == num_batches-1): break
            loss, ans, cont, starts, ends = self.validate_on_batch(sess)
            all_starts.append(starts)
            all_ends.append(ends)
            running_loss +=loss

        all_starts = np.hstack(all_starts)
        all_ends = np.hstack(all_ends)
        f1 = self.get_f1(self.val_set[1], all_starts, all_ends, self.val_set[4])
        avg_loss = running_loss / num_batches
        print('Epoch {} val loss: {:.2E}, f1: {}'.format(epoch, avg_loss, f1))
        print('=========================')
        return avg_loss, f1

    def run_epoch(self, sess, train_examples, epoch):
        num_batches = int(len(train_examples[0]) / self.FLAGS.batch_size)
        running_loss = 0; running_f1 = 0;
        for i in range(num_batches - 1):
            print('Batch {} of {}'.format(i+1, num_batches))
            sess.run(self.tr_inititializer, feed_dict={self.quest_data_placeholder: train_examples[0], self.cont_data_placeholder:train_examples[1], self.ans_data_placeholder:train_examples[2]})

            #if (i == 40): break #
            if (i == num_batches - 1): break #
            loss, beg_logits, end_logits, beg_prob, end_prob, grad_norm, merged  = self.train_on_batch(sess)
            running_loss +=loss
            print('loss: {:.2E}, grad_norm: {}'.format(loss, grad_norm))
            if i % 200 == 0:
                self.write_summaries(merged, epoch, i, num_batches)
                self.write_prob(beg_prob, end_prob)

        avg_loss = running_loss / num_batches

        print('Epoch {} train loss: {:.2E}'.format(epoch, avg_loss))
        print('=========================')

        return avg_loss, grad_norm, beg_prob, end_prob




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
            if epoch == 3:
                pass
                #self.lr *= 10
            epoch_num += 1

            print('')
            logger.info("Training for epoch %d out of %d", epoch_num, epoch_num + self.FLAGS.epochs)
            print('==========')
            tic = time.time()

            tr_loss, grad_norm, beg_prob, end_prob = self.run_epoch(sess, tr_set, epoch)
            print('=========================')
            toc = time.time()
            epoch_dur = (toc - tic) / 60
            logger.info("Epoch took {} minutes".format(epoch_dur))

            logger.info("Validating for epoch %d out of %d", epoch_num, epoch_num + self.FLAGS.epochs)
            print('==========')
            print('')

            val_loss, val_f1 = self.validate(sess, val_set, epoch)
            toc = time.time()
            epoch_dur = (toc - tic) / 60
            self.write_to_train_logs(tr_loss, val_loss, val_f1, epoch_dur)
            logger.info("Epoch took {} minutes".format(epoch_dur))

            #logger.info("Epoch %d out of %d", epoch_num, epoch_num + self.FLAGS.epochs)

            if tr_loss < best_score:
                best_score = tr_loss
                num_since_improve = 0
                self.save_model(best_score, epoch_num, saver, sess)
            else:
                pass
                #self.lr, num_since_improve = self.maybe_change_lr(best_score, num_since_improve)

            '''
            if self.report:
                self.report.log_epoch()
                self.report.save()
            '''
        return best_score


    def train(self, session, tr_set, val_set, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        saver = self.saver
        tic = time.time()
        params = tf.trainable_variables()
        #num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        #logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        self.fit(session, saver, tr_set, val_set, train_dir)
