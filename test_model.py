import argparse

import nltk

from config import TrainingConfig as config
from prepare_data import Variables as v
from prepare_model import Variables as vt
from utils import w2v
import numpy as np


def test(texts=None):
    if texts is None:
        texts = config.TEST_TEXTS
    for text in texts:
        text = text.lower().split()
        word_vectors = [w2v(word, vt.embedder) for word in text]
        postags = nltk.pos_tag(text)
        postags = [t for (w, t) in postags]
        word_postags_encoded = [v.POS2OH[w] for w in postags]

        x_train = np.array(
            [np.concatenate((vec, postag), axis=None) for vec, postag in zip(word_vectors, word_postags_encoded)],
            dtype='float32')
        x_train = np.reshape(x_train, newshape=(1, x_train.shape[0], x_train.shape[1]))
        tags, intent = vt.model.predict_sequence(x_train, len(x_train[0]))
        tags_decoded = [v.INDEX2TAG[np.argmax(pred)] for pred in tags]
        intent_decoded = [v.INDEX2INTENT[np.argmax(intent[0])]]
        print(text)
        print('{}'.format(tags_decoded[:]))
        print('{}'.format(intent_decoded))
        print('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('as')
    parser.add_argument('-t', '--texts',
                        nargs='*',
                        action='store',
                        dest='texts',
                        default=None,
                        help='text data to run the inference')
    res = parser.parse_args()
    print(res)
    test(res.texts)