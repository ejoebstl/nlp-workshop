import numpy as np
import os, sys
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

project_path = '/root'

def get_data(train_size, random_seed=100):
    """ Getting randomly shuffled training / testing data """
    en_text = read_data(os.path.join(project_path, 'data', 'small_vocab_en'))
    fr_text = read_data(os.path.join(project_path, 'data', 'small_vocab_fr'))
    print('Length of text: {}'.format(len(en_text)))

    fr_text = ['sos ' + sent[:-1] + 'eos' if sent.endswith('.') else 'sos ' + sent + ' eos' for sent in fr_text]

    np.random.seed(random_seed)
    inds = np.arange(len(en_text))
    np.random.shuffle(inds)

    train_inds = inds[:train_size]
    test_inds = inds[train_size:]
    tr_en_text = [en_text[ti] for ti in train_inds]
    tr_fr_text = [fr_text[ti] for ti in train_inds]

    ts_en_text = [en_text[ti] for ti in test_inds]
    ts_fr_text = [fr_text[ti] for ti in test_inds]

    print("Average length of an English sentence: {}".format(
        np.mean([len(en_sent.split(" ")) for en_sent in tr_en_text])))
    print("Average length of a French sentence: {}".format(
        np.mean([len(fr_sent.split(" ")) for fr_sent in tr_fr_text])))
    return tr_en_text, tr_fr_text, ts_en_text, ts_fr_text

def read_data(filename):
    """ Reading the zip file to extract text """
    text = []
    with open(filename, 'r', encoding='utf-8') as f:
        i = 0
        for row in f:
            text.append(row)
            i += 1
    return text


def preprocess_data(en_tokenizer, fr_tokenizer, en_text, fr_text, en_timesteps, fr_timesteps):
    """ Preprocessing data and getting a sequence of word indices """

    en_seq = sentence_to_sequence(en_tokenizer, en_text, reverse=False, padding_type='pre', pad_length=en_timesteps)
    fr_seq = sentence_to_sequence(fr_tokenizer, fr_text, pad_length=fr_timesteps)
    print('Vocabulary size (English): {}'.format(np.max(en_seq)+1))
    print('Vocabulary size (French): {}'.format(np.max(fr_seq)+1))
    print('En text shape: {}'.format(en_seq.shape))
    print('Fr text shape: {}'.format(fr_seq.shape))

    return en_seq, fr_seq

def sentence_to_sequence(tokenizer, sentences, reverse=False, pad_length=None, padding_type='post'):
    encoded_text = tokenizer.texts_to_sequences(sentences)
    preproc_text = pad_sequences(encoded_text, padding=padding_type, maxlen=pad_length)
    if reverse:
        preproc_text = np.flip(preproc_text, axis=1)

    return preproc_text