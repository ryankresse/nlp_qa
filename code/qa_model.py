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
                'logits_weight': tf.get_variable('logits_weight',shape=[self.FLAGS.state_size * 4], dtype=tf.float64),
                'attention_weight': tf.get_variable('attention_weight', shape=[self.FLAGS.state_size * 4, self.FLAGS.state_size * 2], dtype=tf.float64)
                }

    def add_biases(self):
        with tf.variable_scope('biases') as scope:
            self.biases = {
                'logits_bias': tf.get_variable('logits_bias', shape=[1], dtype=tf.float64)
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
        self.ans_placeholder = tf.placeholder(tf.float64, shape=(None, self.FLAGS.cont_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(None))



    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        self.pred = self.add_prediction_op()
        self.loss = self.get_loss(self.pred, self.ans_placeholder)
        self.train_op, self.grad_norm = self.add_train_op(self.loss)

    def get_loss(self, logits, labels):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            example_sum_loss = tf.reduce_sum(losses, axis=1)
            return tf.reduce_mean(example_sum_loss)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            pass

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
        optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)
        gradients, var = zip(*optimizer.compute_gradients(loss))
        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient_norm)

        train_op = optimizer.apply_gradients(zip(gradients, var))
        return train_op, grad_norm  # tf.global_norm(gradients)


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

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

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
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size)
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size)
            output, hidden_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, quest_embed, dtype=tf.float64)

            fw_c = hidden_state[0].c
            bw_c = hidden_state[1].c
            fw_h = hidden_state[0].h
            bw_h = hidden_state[1].h


            last_hidden_state_tuple = tf.contrib.rnn.LSTMStateTuple(tf.concat([fw_c, bw_c], axis=1), tf.concat([fw_h, bw_h], axis=1))
            last_hidden_state = tf.concat([fw_h, bw_h], axis=1)
            return output, last_hidden_state_tuple, last_hidden_state
            #return tf.concat([fw_output, bw_output], axis=2), tf.concat([fw_hidden_state, bw_hidden_state], axis=1)

    def get_cont_rep(self, cont_embed, quest_last_hid):
        with tf.variable_scope('cont_rep_rnn') as scope:
            bw_cont_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size * 2)
            fw_cont_cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size * 2)
            output, hidden_state = tf.nn.bidirectional_dynamic_rnn(fw_cont_cell, bw_cont_cell, cont_embed, initial_state_fw=quest_last_hid, initial_state_bw=quest_last_hid)
            return tf.concat([output[0], output[1]], axis=2)

    def final_lstm(self, cont_scaled):
        with tf.variable_scope('final_lstm') as scope:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.state_size * 4)
            output, hidden_state = tf.nn.dynamic_rnn(cell, cont_scaled, dtype=tf.float64)
            return output


    def get_logits(self, raw_output):
        #(max_time, state_size*4)
        with tf.variable_scope('logits', reuse=True) as scope:
            logits = []
            for t in np.arange(self.FLAGS.cont_length):
                time_step = tf.squeeze(tf.slice(raw_output, [0,t,0], [-1,1,-1])) #(batch,embed)
                multiplied = time_step * self.weights['logits_weight']
                summed = tf.reduce_sum(multiplied, axis=1) + self.biases['logits_bias'] #(batch, cont_length)
                logits.append(summed)
            logits_concat = tf.stack(logits, axis=1)
            return logits_concat

    def calculate_att_vectors(self, cont_hid, quest_hid):
        with tf.variable_scope('attention', reuse=True) as scope:
            all_scores = []
            for example in np.arange(self.FLAGS.batch_size):
                scores = []
                quest_hid_for_example = tf.slice(quest_hid, [example, 0], [1, -1])
                for time in np.arange(self.FLAGS.cont_length):
                    hidden_cont = tf.slice(cont_hid, [example, time, 0], [1,1,-1]) #(1, embed*4)
                    hidden_cont = tf.squeeze(hidden_cont, axis=0)
                    intermediate =  tf.matmul(hidden_cont, self.weights['attention_weight'])
                    scores.append(tf.matmul(intermediate, tf.transpose(quest_hid_for_example)))
                all_scores.append(scores)

            squeezed = tf.squeeze(tf.stack(all_scores))
            return tf.nn.softmax(squeezed)

    def scale_cont(self, cont, att):
        scaled = []
        for example in np.arange(self.FLAGS.batch_size):
            ex = tf.slice(cont,[example, 0, 0], [1, -1, -1])
            ex = tf.squeeze(ex)
            att_vec = tf.slice(att, [example,0], [1, -1])
            att_mat = tf.tile(att_vec, [self.FLAGS.state_size*4, 1])
            scaled.append(ex * tf.transpose(att_mat))
        return tf.stack(scaled)

    def add_prediction_op(self):
        quest_embed, cont_embed = self.add_embeddings()
        quest_out, quest_last_hid_tuple, quest_last_hid = self.get_quest_rep(quest_embed) #output(batch, max_time, hidden*2), last_hidden_state(batch, hidden*2)
        #return quest_out, quest_last_hid_tuple, quest_last_hid
        cont_out = self.get_cont_rep(cont_embed, quest_last_hid_tuple) #(batch, max_time, embed*4)
        att_vectors = self.calculate_att_vectors(cont_out, quest_last_hid)
        scaled_cont = self.scale_cont(cont_out, att_vectors)
        final_hid = self.final_lstm(scaled_cont)
        logits = self.get_logits(final_hid)
        return logits
        #quest_out is a tuple of fw_output and bw_output. Each is shaped (batch, max_time, output_size)
        #Each item in this tuple is a LSTMStateTuple, consisting of the cell state and the hidden state Each is of shape (batch, output_size)
        #Next, run the bilstm on the context to get the outputs for that one.



    def train_on_batch(self, sess, quest_batch, cont_batch, ans_batch):
        feed = self.create_feed_dict(quest_batch, cont_batch, ans_batch, self.FLAGS.dropout)
        #_, loss, logits, gradient_global_norm = sess.run([self.train_op, self.loss, self.pred, self.gradient_global_norm], feed_dict=feed)
        logits = sess.run(self.pred, feed_dict=feed)
        _, loss, logits, grad_norm = sess.run([self.train_op, self.loss, self.pred, self.grad_norm], feed_dict=feed)
        #return att
        return loss, logits, grad_norm

    def get_ans_words(self, logits, truth, cont_text, cont_length):
        probs = sigmoid(logits)
        #pdb.set_trace()
        pred_ans_bools = probs >  0.5
        ans_text = []
        pred_text = []
        for ans, cont, bools in zip(truth, cont_text, pred_ans_bools):
            idx = list(np.nonzero(ans)[0])
            cont_arr = cont.strip().split(" ")
            pad_needed = cont_length - max(len(cont_arr), 0)
            cont_arr += ['garblegarble'] * pad_needed
            ans_words = np.array(cont_arr)[idx]
            ans_words = " ".join(list(ans_words))
            ans_text.append(ans_words)
            pred_idx = np.where(bools)
            pred_words = np.array(cont_arr)[pred_idx]
            pred_words = " ".join(list(pred_words))
            pred_text.append(pred_words)

        return pred_text, ans_text


    def evaluate_performance(self, pred_ans_words, true_ans_words, ans_text):
        f1 =  get_f1_score(pred_ans_words, true_ans_words)
        return f1


    def run_epoch(self, sess, train_examples):
        prog = Progbar(target=1 + int(len(train_examples[0]) / self.FLAGS.batch_size))
        num_batches = int(len(train_examples[0]) / self.FLAGS.batch_size)
        for i, batch in enumerate(minibatches(train_examples, self.FLAGS.batch_size)):
            print('Batch {} of {}'.format(i, num_batches))
            quest = batch[0]; cont = batch[1]; ans = batch[2]; cont_text = batch[3]; ans_text = batch[4];
            loss, logits, grad_norm = self.train_on_batch(sess, quest, cont, ans)
            #pdb.set_trace()
            #print('batch {}, gradient_global_norm: {}'.format(i, gradient_global_norm))
            if (i+1) % 1 == 0:
                pred_ans_words, true_ans_words = self.get_ans_words(logits, ans, cont_text, self.FLAGS.cont_length)
                f1 = self.evaluate_performance(pred_ans_words, true_ans_words, ans_text)
                print('batch {}, loss: {}, f1: {}, grad_norm: {}'.format(i, loss, f1, grad_norm))
                #prog.update(i + 1, [("train loss", loss)])

            #if self.report: self.report.log_train_loss(loss)
        print("")

    def fit(self, sess, saver, train_data, train_dir):
        best_score = 0.

        for epoch in range(self.FLAGS.epochs):
            score = self.run_epoch(sess, train_data)
            logger.info("Epoch %d out of %d", epoch + 1, self.FLAGS.epochs)
            '''
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score
        '''

    def train(self, session, dataset, train_dir):
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
        blank = tf.Variable(tf.zeros((3,3)), name='blank')
        saver = tf.train.Saver()
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        self.fit(session, saver, dataset, train_dir)
