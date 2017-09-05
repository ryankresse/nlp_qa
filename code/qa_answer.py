from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin
import pdb
from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf
import data_utils
import pickle

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS
MODEL_NAME= 'multi_35'
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_string("beg_prob_file", 'beg_prob.npy', "File to beg write probabilities")
tf.app.flags.DEFINE_string("summaries_dir", 'summaries_dir', "Folder for summaries")
tf.app.flags.DEFINE_string("end_prob_file", 'end_prob.npy', "File to end write probabilities")
tf.app.flags.DEFINE_integer("num_epochs_per_anneal", 5, "The learning rate will be annealed if the model doesn't improve in this many epochs.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 70, "Batch size to use during training.")
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
tf.app.flags.DEFINE_integer("num_perspectives", 35, "The number of matching perspectives")




def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
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


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []
    context_text = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                '''context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                '''

                context_ids =[int(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [int(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]
                context_data.append(context_ids)
                query_data.append(qustion_ids)

                question_uuid_data.append(question_uuid)
                context_text.append(context_tokens)
    query_data = data_utils.clip_and_pad(query_data, FLAGS.quest_length, FLAGS.pad_token)
    context_data = data_utils.clip_and_pad(context_data, FLAGS.cont_length, FLAGS.pad_token)
    context_text =  data_utils.clip_and_pad(context_text, FLAGS.cont_length, FLAGS.pad_token)
    return context_data, query_data, question_uuid_data, context_text


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))

    context_data, question_data, question_uuid_data, context_text = read_dataset(dev_data, 'dev', vocab)
    #for i in range(len(context_data)):
    #context_data = [int(t) for t in l in context_data in l.split()]

    return context_data, question_data, question_uuid_data, context_text

def get_ans_words(starts_ends, cont, idx_word):
    words = []

    for ix in np.arange(starts_ends.shape[0]):

        start = starts_ends[ix, 0]; end = starts_ends[ix,1];
        if start > end:
            words.append([])
        elif start == end:
            #select the token from the training example at the start position, then get the word for it
            #words.append([idx_word[cont[ix, start]]])
            words.append(cont[ix, start])
        else:
            ws = cont[ix, start:end + 1]
            words.append(ws)
            #words.append([idx_word[tok] for tok in tokens])

    return [" ".join(w) for w in words]

def generate_answers(sess, model, dataset, rev_vocab, context_text, idx_word):
    all_starts =[]; all_ends = [];
    starts, ends = model.test(sess, dataset)
    pred_words = get_ans_words(np.hstack([np.expand_dims(starts,1), np.expand_dims(ends,1)]), context_text, idx_word)
    #f1 = self.evaluate_performance(pred_words, ans_text)

    answers = {}
    for uid, words in zip(dataset[2], pred_words):
        answers[uid] = words
    with open('answers.pkl', 'w') as f:
        pickle.dump(answers, f)

    return answers


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

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, question_uuid_data, context_text = prepare_dev(dev_dirname, dev_filename, vocab)
    #pdb.set_trace()
    dataset = (context_data, question_data, question_uuid_data)
    #so you need you model to have an evaluation mode.
    #use the same setup as validation.


    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    idx_word = data_utils.invert_map(vocab)

    qa = QASystem(FLAGS, embed_path, idx_word, False, 0, True)
    with tf.Session() as sess:
        initialize_model(sess, qa, FLAGS.train_dir)
        #start, end = qa.test(sess, dataset)
        #pdb.set_trace()
        answers = generate_answers(sess, qa, dataset, rev_vocab, context_text, idx_word)
        #with open('answers.pkl', 'r') as f:
        #    ans = pickle.load(f)
        #pdb.set_trace()
        #d = json.dumps(ans,  ensure_ascii=False, encoding='latin-1')
        #pdb.set_trace()
        # write to json file to root dir

        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False, encoding='latin-1')))

if __name__ == "__main__":
  tf.app.run()
