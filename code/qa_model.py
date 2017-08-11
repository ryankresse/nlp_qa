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
        self.pretrained_embeddings = np.load(embed_path, mmap_mode='r')['glove']

        # ==== set up placeholder tokens ========
        self.cont_placeholer = None
        self.quest_placeholder = None
        self.ans_placeholder = None
        self.dropout_placeholder = None


        self.add_placeholders()
        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            #self.setup_embeddings()
            #self.set_add_prediction_op()
            self.add_weights()
            self.add_biases()
            self.setup_system()

            #self.setup_loss()
        # ==== set up training/updating procedure ====
        pass

    def add_weights(self):
        #out is (batch, quest_length, hidden_size*2)
        with tf.variable_scope('weights') as scope:
            self.weights = {
                'beg_mlp_weight1': tf.get_variable('beg_mlp_weight1',shape=[self.FLAGS.state_size*2, 1], dtype=tf.float64),
                'end_mlp_weight1': tf.get_variable('end_mlp_weight1',shape=[self.FLAGS.state_size*2, 1], dtype=tf.float64),
                'beg_mlp_weight2': tf.get_variable('beg_mlp_weight2',shape=[self.FLAGS.quest_length + self.FLAGS.cont_length, self.FLAGS.cont_length], dtype=tf.float64),
                'end_mlp_weight2': tf.get_variable('end_mlp_weight2',shape=[self.FLAGS.quest_length + self.FLAGS.cont_length, self.FLAGS.cont_length], dtype=tf.float64),
                'attention_weight': tf.get_variable('attention_weight', shape=[self.FLAGS.state_size*4, self.FLAGS.state_size*2], dtype=tf.float64)
                }

    def add_biases(self):
        with tf.variable_scope('biases') as scope:
            self.biases = {
                'beg_logits_bias': tf.get_variable('beg_logits_bias', shape=[self.FLAGS.state_size], dtype=tf.float64),
                'end_logits_bias': tf.get_variable('end_logits_bias', shape=[self.FLAGS.state_size], dtype=tf.float64)
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



    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        beg_logits, end_logits = self.add_prediction_op()
        self.beg_labels, self.end_labels = self.get_labels(self.ans_placeholder)
        self.loss = self.get_loss(beg_logits, end_logits, self.beg_labels, self.end_labels)
        tf.summary.scalar('loss', self.loss)
        self.train_op, self.grad_norm = self.add_train_op(self.loss)
        tf.summary.scalar('grad norm', self.grad_norm)

        self.beg_prob = tf.nn.softmax(self.beg_logits)
        self.end_prob = tf.nn.softmax(self.end_logits)
        tf.summary.histogram('beg_prob', self.beg_prob)
        tf.summary.histogram('end_prob', self.end_prob)

        self.starts = self.get_pred(self.end_prob)
        self.ends = self.get_pred(self.end_prob)
        tf.summary.histogram('ans_len', self.ends - self.starts)
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.FLAGS.summaries_dir)
        self.saver = tf.train.Saver()

    def apply_mask(self, items):
        items_flat = tf.reshape(items, [-1])
        cont_flat = tf.reshape(self.cont_placeholder, [-1])
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
            quest_embed = tf.nn.embedding_lookup(embeddings, self.quest_placeholder)
            cont_embed = tf.nn.embedding_lookup(embeddings, self.cont_placeholder)
        return (quest_embed, cont_embed)

    def add_train_op(self, loss):
        optimizer = tf.train.RMSPropOptimizer(self.FLAGS.learning_rate)
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


    def get_quest_rep(self, quest_embed):
        with tf.variable_scope('quest_rep_rnn') as scope:
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size, activation=tf.nn.relu)
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size, activation=tf.nn.relu)
            output, hidden_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, quest_embed, sequence_length=self.quest_lens, dtype=tf.float64)

            fw_c = hidden_state[0].c
            bw_c = hidden_state[1].c
            fw_h = hidden_state[0].h
            bw_h = hidden_state[1].h


            last_hidden_state_tuple = tf.contrib.rnn.LSTMStateTuple(fw_c +  bw_c, fw_h + bw_h)
            last_hidden_state = tf.concat([fw_h, bw_h], 1)
            return tf.concat(output, 2), last_hidden_state_tuple, last_hidden_state
            #return tf.concat([fw_output, bw_output], axis=2), tf.concat([fw_hidden_state, bw_hidden_state], axis=1)

    def get_cont_rep(self, cont_embed, quest_hid_state):
        with tf.variable_scope('cont_rep_rnn') as scope:
            bw_cont_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size, activation=tf.nn.relu)
            fw_cont_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size, activation=tf.nn.relu)
            output, hidden_state = tf.nn.bidirectional_dynamic_rnn(fw_cont_cell, bw_cont_cell, cont_embed, initial_state_fw=quest_hid_state, initial_state_bw=quest_hid_state, sequence_length=self.cont_lens)
            return tf.concat(output, 2)

    def beg_lstm(self, cont_scaled):
        with tf.variable_scope('beg_lstm') as scope:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size, activation=tf.nn.relu)
            output, hidden_state = tf.nn.dynamic_rnn(cell, cont_scaled, dtype=tf.float64, sequence_length=self.cont_lens)
            return output

    def end_lstm(self, cont_scaled):
        with tf.variable_scope('end_lstm') as scope:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size, activation=tf.nn.relu)
            output, hidden_state = tf.nn.dynamic_rnn(cell, cont_scaled, dtype=tf.float64, sequence_length=self.cont_lens)
            return output

    def get_beg_logits(self, raw_output):
        #(max_time, state_size*4)
        with tf.variable_scope('beg_logits') as scope:
            logits = []
            for example in np.arange(self.FLAGS.batch_size):
                ex = tf.squeeze(tf.slice(raw_output, [example, 0, 0], [1, -1,-1]))
                hidden = tf.nn.relu(tf.matmul(ex, self.weights['beg_mlp_weight1']))
                out = tf.matmul(tf.transpose(hidden), self.weights['beg_mlp_weight2'])
                logits.append(out)
            return tf.squeeze(tf.stack(logits))

    def get_end_logits(self, raw_output):
        #(max_time, state_size*4)
        with tf.variable_scope('end_logits') as scope:
            logits = []
            for example in np.arange(self.FLAGS.batch_size):
                ex = tf.squeeze(tf.slice(raw_output, [example, 0, 0], [1, -1,-1]))
                hidden = tf.nn.relu(tf.matmul(ex, self.weights['end_mlp_weight1']))
                out = tf.matmul(tf.transpose(hidden), self.weights['end_mlp_weight2'])
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

    def calculate_att_vectors(self, quest_last_hid, cont_hid):
        with tf.variable_scope('attention') as scope:
            all_scores = []
            for example in np.arange(self.FLAGS.batch_size):
                quest_hid_for_example = tf.slice(quest_last_hid, [example, 0], [1, -1]) # (400,1)
                ex_cont = tf.slice(cont_hid, [example, 0, 0], [1,-1,-1])
                ex_cont = tf.squeeze(ex_cont) #(300, 400) (400, 400)
                all_scores.append(tf.matmul(ex_cont, tf.transpose(quest_hid_for_example)))
            return tf.nn.softmax(tf.squeeze(tf.stack(all_scores)))

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
        quest_mask = tf.sign(self.quest_placeholder)
        cont_mask = tf.sign(self.cont_placeholder)
        quest_lens = tf.reduce_sum(quest_mask,axis=1)
        cont_lens = tf.reduce_sum(cont_mask ,axis=1)
        return quest_lens, cont_lens

    def get_labels(self, labels):
        labels = self.clip_labels(labels)
        beg_labels = tf.slice(labels, [0, 0], [-1,1])
        end_labels = tf.slice(labels, [0, 1], [-1,1])
        return tf.squeeze(beg_labels), tf.squeeze(end_labels)

    def add_prediction_op(self):
        self.quest_lens, self.cont_lens = self.get_lens()

        quest_embed, cont_embed = self.add_embeddings()
        quest_out, quest_last_hid_tuple, quest_last_hid = self.get_quest_rep(quest_embed) #output(batch, max_time, hidden*2), last_hidden_state(batch, hidden*2)
        self.quest_last_hid = quest_last_hid
        self.quest_out = quest_out
        cont_out = self.get_cont_rep(cont_embed, quest_last_hid_tuple)
        self.cont_out = cont_out
        self.att_vectors = self.calculate_att_vectors(quest_last_hid, cont_out)
        self.scaled_cont = self.scale_cont(cont_out, self.att_vectors)
        self.scaled_cont_concat = tf.concat([self.scaled_cont, self.cont_out], axis=2)
        self.scaled_cont_concat = self.transform_scaled_cont_concat(self.scaled_cont_concat)

        self.out_concat = tf.concat([self.quest_out, self.scaled_cont_concat], axis=1)
        self.beg_logits = self.apply_mask(self.get_beg_logits(self.out_concat))
        self.end_logits = self.apply_mask(self.get_end_logits(self.out_concat))
        return self.beg_logits, self.end_logits



    def debug_on_batch(self, sess, quest_batch, cont_batch, ans_batch):
        feed = self.create_feed_dict(quest_batch, cont_batch, ans_batch, self.FLAGS.dropout)
        quest_last_hid, cont_out = sess.run([self.self.quest_last_hid, self.cont_out], feed_dict=feed)
        _, loss, logits, grad_norm = sess.run([self.train_op, self.loss, self.pred, self.grad_norm], feed_dict=feed)
        return loss, logits, grad_norm

    def get_ans_words(self, starts_ends, cont):
        words = []
        for ix in np.arange(starts_ends.shape[0]):
            start = starts_ends[ix, 0]; end = starts_ends[ix,1];
            if start > end:
                words.append([])
            elif start == end:
                words.append([self.idx_word[cont[ix, start]]])
            else:
                tokens = cont[ix, start:end + 1]
                words.append([self.idx_word[tok] for tok in tokens])

        return [" ".join(w) for w in words]


    def get_pred(self, probs):
        return tf.argmax(probs, axis=1)

    def evaluate_performance(self, pred_ans_words, true_ans_words, ans_text):
        f1 =  get_f1_score(pred_ans_words, true_ans_words)
        return f1


    def debug_epoch(self, sess, train_examples, num_batches):
        print('DEBUGGING')
        for i, batch in enumerate(minibatches(train_examples, self.FLAGS.batch_size)):
            print('Batch {} of {}'.format(i, num_batches))
            quest = batch[0]; cont = batch[1]; ans = batch[2]; cont_text = batch[3]; ans_text = batch[4];

            loss, logits, grad_norm = self.train_on_batch(sess, quest, cont, ans)
            if (i+1) % 5 == 0:
                pred_ans_words, true_ans_words = self.get_ans_words(logits, ans, cont_text, self.FLAGS.cont_length)
                words_pred = np.sum([len(ans.split()) for ans in pred_ans_words])
                f1 = self.evaluate_performance(pred_ans_words, true_ans_words, ans_text)
                print('batch {}, loss: {}, f1: {}, grad_norm: {}, words predicted: {}'.format(i, loss, f1, grad_norm, words_pred))

    def write_summaries(self, summaries, epoch, batch, num_batches):
        self.summary_writer.add_summary(summaries, (epoch * num_batches) + batch)

    def get_f1(self, ans, cont, starts, ends, ans_text):
        true_words = self.get_ans_words(ans, cont)
        pred_words = self.get_ans_words(np.hstack([np.expand_dims(starts,1), np.expand_dims(ends,1)]), cont)
        f1 = self.evaluate_performance(pred_words, true_words, ans_text)
        #print('f1: {}'.format(f1))
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

    def train_on_batch(self, sess, quest_batch, cont_batch, ans_batch):
        feed = self.create_feed_dict(quest_batch, cont_batch, ans_batch, self.FLAGS.dropout)
        train_op, loss, beg_logits, end_logits, beg_prob, end_prob, grad_norm, clip_value, starts, ends, merged =  sess.run([self.train_op, self.loss, self.beg_logits, self.end_logits, self.beg_prob, self.end_prob, self.grad_norm, self.clip_val, self.starts, self.ends, self.merged], feed_dict=feed)
        return loss, beg_logits, end_logits, beg_prob, end_prob, starts, ends, grad_norm, clip_value, merged

    def validate_on_batch(self, sess, quest_batch, cont_batch, ans_batch):
        feed = self.create_feed_dict(quest_batch, cont_batch, ans_batch, self.FLAGS.dropout)
        loss, starts, ends =  sess.run([self.loss, self.starts, self.ends], feed_dict=feed)
        return loss, starts, ends

    def run_epoch(self, sess, train_examples, epoch):
        num_batches = int(len(train_examples[0]) / self.FLAGS.batch_size)
        print('Training for epoch num: {}'.format(epoch))
        print('==========')
        print('')
        running_loss = 0; running_f1 = 0;
        for i, batch in enumerate(minibatches(train_examples, self.FLAGS.batch_size)):
            print('Batch {} of {}'.format(i+1, num_batches))
            if (i == num_batches - 1): break #
            quest = batch[0]; cont = batch[1]; ans = batch[2]; cont_text = batch[3]; ans_text = batch[4]; quest_text=batch[5];
            loss, beg_logits, end_logits, beg_prob, end_prob, starts, ends, grad_norm, clip_value, merged  = self.train_on_batch(sess, quest, cont, ans)
            running_loss +=loss
            print('loss: {:.2E}, grad_norm: {}, clip_value: {}'.format(loss, grad_norm, clip_value))
            self.write_prob(beg_prob, end_prob)
            self.write_summaries(merged, epoch, i, num_batches)
            running_f1 += self.get_f1(ans, cont, starts, ends, ans_text)

        avg_loss, avg_f1 = self.compute_and_report_epoch_stats(epoch, running_loss, running_f1, num_batches)
        return avg_loss, avg_f1, grad_norm, clip_value, beg_prob, end_prob

    def validate(self, sess, val_set, epoch):
        print('Running Validation for epoch: {}'.format(epoch))
        print('==========')
        print('')
        num_batches = int(len(val_set[0]) / self.FLAGS.batch_size)
        running_loss = 0; running_f1 = 0;
        for i, batch in enumerate(minibatches(val_set, self.FLAGS.batch_size)):
            print('Batch {} of {}'.format(i+1, num_batches))
            if (i == num_batches-1): break
            quest = batch[0]; cont = batch[1]; ans = batch[2]; cont_text = batch[3]; ans_text = batch[4]; quest_text=batch[5];
            loss, starts, ends  = self.validate_on_batch(sess, quest, cont, ans)
            running_loss +=loss
            running_f1 += self.get_f1(ans, cont, starts, ends, ans_text)

        avg_loss, avg_f1 = self.compute_and_report_epoch_stats(epoch, running_loss, running_f1, num_batches, 'val')
        #self.write_val_summaries(sess, epoch, avg_loss, avg_f1)
        return avg_loss, avg_f1


    def retrieve_prev_best_score(self):
        with open(self.FLAGS.prev_best_score_file, 'r') as the_file:
            best_score = the_file.readline()
        if best_score == '':
            print('no previous best score found. setting best score to zero')
            best_score = 0.
        else:
            print('restored previous best score of {}'.format(best_score))
            best_score = float(best_score)
        print('=========')
        print('')
        return best_score

    def write_to_train_logs(self, f1, loss, grad_norm, clip_value, beg_prob, end_prob, val_loss, val_f1):
        with open(self.FLAGS.train_stats_file, 'a') as f:
            max_beg_prob = np.mean(np.max(beg_prob, axis=1))
            max_end_prob = np.mean(np.max(end_prob, axis=1))
            f.write("{},{},{},{},{},{},{},{}\n".format(f1, loss, grad_norm, clip_value, max_beg_prob, max_end_prob,val_loss, val_f1))

    def maybe_save_model_and_change_best_score(self, f1, best_score, saver, sess):
        if f1 > best_score:
            best_score = f1
            if saver:
                logger.info("New best score! Saving model.")
                saver.save(sess, self.FLAGS.train_dir+'/' + self.FLAGS.ckpt_file_name)
                with open('best_score.txt', 'w') as the_file:
                    the_file.write(str(best_score))
        else:
            print("f1 didn't improve on best score of {}. not saving model.".format(best_score))
        print('==========')
        print('')
        return best_score

    def write_val_summaries(self, sess, epoch, val_loss, val_f1):
        val_loss_tensor = tf.constant(val_loss)
        val_f1_tensor = tf.constant(val_f1)
        val_loss_summ = tf.summary.scalar('val_loss',val_loss_tensor)
        val_f1_summ = tf.summary.scalar('val_f1', val_f1_tensor)
        val_merged = tf.summary.merge([val_loss_summ, val_f1_summ])
        merged_for_write = sess.run(val_merged)
        self.summary_writer.add_summary(merged_for_write, epoch)



    def fit(self, sess, saver, tr_set, val_set, train_dir):
        best_score = self.retrieve_prev_best_score()

        for epoch in range(self.FLAGS.epochs):
            tr_loss, tr_f1, grad_norm, clip_value, beg_prob, end_prob = self.run_epoch(sess, tr_set, epoch)

            #self.write_to_train_logs(tr_f1, tr_loss, grad_norm, clip_value, beg_prob, end_prob)
            val_loss, val_f1 = self.validate(sess, val_set, epoch)
            self.write_to_train_logs(tr_f1, tr_loss, grad_norm, clip_value, beg_prob, end_prob, val_loss, val_f1)


            logger.info("Epoch %d out of %d", epoch + 1, self.FLAGS.epochs)
            best_score = self.maybe_save_model_and_change_best_score(val_f1, best_score, saver, sess)


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
        #tic = time.time()
        #params = tf.trainable_variables()
        #num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        #toc = time.time()
        #logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        self.fit(session, saver, tr_set, val_set, train_dir)
