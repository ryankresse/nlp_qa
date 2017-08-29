from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import pdb
import tensorflow as tf
import sys
import numpy as np
from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
import data_utils
import shutil
import logging

logging.basicConfig(level=logging.INFO)
MODEL_NAME= 'mult_per_match_32'
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_string("beg_prob_file", 'beg_prob.npy', "File to beg write probabilities")
tf.app.flags.DEFINE_string("summaries_dir", 'summaries_dir', "Folder for summaries")
tf.app.flags.DEFINE_string("end_prob_file", 'end_prob.npy', "File to end write probabilities")
tf.app.flags.DEFINE_integer("num_epochs_per_anneal", 5, "The learning rate will be annealed if the model doesn't improve in this many epochs.")


tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 40, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("prev_best_score_file", "best_score.txt", "File where previous best score for model is stored")
tf.app.flags.DEFINE_string("train_stats_file", "train_logs.txt", "File to store stats for the model")

tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train_dir/{}".format(MODEL_NAME), "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "train_dir/".format(MODEL_NAME), "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("ckpt_file_name", MODEL_NAME, "Checkpoint file name")
tf.app.flags.DEFINE_string("sample_data_prepend", "samp.", "String prepended to data file to indicate it contains a small sample of the original data set")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_integer("pad_token", 0, "Token be used when padding data to be the same length")
tf.app.flags.DEFINE_integer("cont_length", 250, "The length the context should be padded or clipped to so that the model receives inputs of uniform length")
tf.app.flags.DEFINE_integer("quest_length", 25, "The length the question should be padded or clipped to so that the model receives inputs of uniform length")
tf.app.flags.DEFINE_integer("ans_length", 2, "The length of the answer")
tf.app.flags.DEFINE_integer("num_perspectives", 30, "The number of matching perspectives")


FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    #restore the model from checkpoint if it is already exits. Else initialize the variables and return the model
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        with open(FLAGS.prev_best_score_file, 'w') as f: #clear any best score from other models
            f.write('')
        shutil.rmtree(FLAGS.summaries_dir) # remove summarries from previous model
        os.makedirs(FLAGS.summaries_dir)
        os.remove(FLAGS.train_stats_file)
        os.mknod(FLAGS.train_stats_file)

        #logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def fetch_data_set(prepend, set_name):
    context_data = data_utils.read_clip_and_pad(pjoin(FLAGS.data_dir, prepend+'{}.ids.context'.format(set_name)), FLAGS.cont_length, FLAGS.pad_token)
    question_data = data_utils.read_clip_and_pad(pjoin(FLAGS.data_dir, prepend+'{}.ids.question'.format(set_name)), FLAGS.quest_length, FLAGS.pad_token)
    answer_data = np.array(data_utils.read_token_data_file(pjoin(FLAGS.data_dir, prepend+'{}.span'.format(set_name))), dtype=np.int32)

    #for producing F1 and EM scores
    context_text = data_utils.read_text_data_file(pjoin(FLAGS.data_dir, prepend+'{}.context'.format(set_name)))
    quest_text = data_utils.read_text_data_file(pjoin(FLAGS.data_dir, prepend+'{}.question'.format(set_name)))
    ans_text = data_utils.read_text_data_file(pjoin(FLAGS.data_dir, prepend+'{}.answer'.format(set_name)))
    print('Finished reading {} set of {} examples'.format(set_name, context_data.shape[0]))
    return [question_data, context_data, answer_data, context_text, ans_text, quest_text]

def test_device_placement():
    print('Running device placement test')
    print('=======')
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as test_sess:
        test_mat_mul = tf.matmul(tf.ones([4,4]), tf.zeros([4,4]))
        print(test_sess.run(test_mat_mul))

def main(_):
    test_device_placement()
    # if the user doesn't pass in 'train' on the command line, we're just going to use a small subest of the train data
    prepend = '' #  if len(sys.argv) > 1 and sys.argv[1] == 'train' else FLAGS.sample_data_prepend


    print('Reading data')
    print('==================')

    tr_set = fetch_data_set(prepend, 'train')
    val_set = fetch_data_set(prepend, 'val')
    print('Finished reading data')
    print('==================')


    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")

    #vocab is map from words to indices, rev_vocab is our list of words in reverse frequency order
    vocab, rev_vocab = initialize_vocab(vocab_path)

    idx_word = data_utils.invert_map(vocab)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder, FLAGS, embed_path, idx_word, tr_set, val_set)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    #print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)


    with tf.Session() as sess:
        #load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, FLAGS.train_dir)
        #save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, tr_set, val_set, FLAGS.train_dir)

        qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
