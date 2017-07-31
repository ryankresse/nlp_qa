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

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train_dir/tiny_samp", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "train_dir/tiny_samp", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("ckpt_file_name", "tiny_samp", "Checkpoint file name")
tf.app.flags.DEFINE_string("sample_data_prepend", "tiny.samp.", "String prepended to data file to indicate it contains a small sample of the original data set")

tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_integer("pad_token", 0, "Token be used when padding data to be the same length")
tf.app.flags.DEFINE_integer("cont_length", 300, "The length the context should be padded or clipped to so that the model receives inputs of uniform length")
tf.app.flags.DEFINE_integer("quest_length", 45, "The length the question should be padded or clipped to so that the model receives inputs of uniform length")
tf.app.flags.DEFINE_integer("ans_length", 2, "The length of the answer")


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
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
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


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = None

    # if the user doesn't pass in 'train' on the command line, we're just going to use a small subest of the train data
    prepend = '' if len(sys.argv) > 1 and sys.argv[1] == 'train' else FLAGS.sample_data_prepend

    print('Reading data')
    print('==================')


    context_data = data_utils.read_clip_and_pad(pjoin(FLAGS.data_dir, prepend+'train.ids.context'), FLAGS.cont_length, FLAGS.pad_token)
    question_data = data_utils.read_clip_and_pad(pjoin(FLAGS.data_dir, prepend+'train.ids.question'), FLAGS.quest_length, FLAGS.pad_token)
    answer_data = np.array(data_utils.read_token_data_file(pjoin(FLAGS.data_dir, prepend+'train.span')), dtype=np.int32)
    dense_answers = data_utils.make_dense_answers(answer_data, FLAGS.cont_length)

    #for producing F1 and EM scores
    context_text = data_utils.read_text_data_file(pjoin(FLAGS.data_dir, prepend+'train.context'))

    #for debugging purposes. Remove when model is training properly
    ans_text = data_utils.read_text_data_file(pjoin(FLAGS.data_dir, prepend+'train.answer'))


    dataset = [question_data, context_data, dense_answers, context_text, ans_text]
    print('Finished reading data')


    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")

    #vocab is map from words to indices, rev_vocab is our list of words in reverse frequency order
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder, FLAGS, embed_path)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        #load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, FLAGS.train_dir)
        #save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, FLAGS.train_dir)

        qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
