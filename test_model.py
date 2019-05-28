import argparse

import gensim
import nltk
import numpy as np

from config import TrainingConfig as config
from models.seq2seq import EncoderDecoder
from prepare_data import load_dicts
from utils import w2v


class Variables:
    POS2OH = None
    INTENT2OH = None
    INDEX2INTENT = None
    TAG2OH = None
    INDEX2TAG = None


def test(model, embedder, texts=None):
    if texts is None:
        texts = config.TEST_TEXTS
    for text in texts:
        text = text.lower().split()
        word_vectors = [w2v(word, embedder) for word in text]
        postags = nltk.pos_tag(text)
        postags = [t for (w, t) in postags]
        word_postags_encoded = [Variables.POS2OH[w] for w in postags]

        x_train = np.array(
            [np.concatenate((vec, postag), axis=None) for vec, postag in zip(word_vectors, word_postags_encoded)],
            dtype='float32')
        x_train = np.reshape(x_train, newshape=(1, x_train.shape[0], x_train.shape[1]))
        tags, intent = model.predict_sequence(x_train, len(x_train[0]))
        tags_decoded = [Variables.INDEX2TAG[np.argmax(pred)] for pred in tags]
        intent_decoded = [Variables.INDEX2INTENT[np.argmax(intent[0])]]
        print(text)
        print('{}'.format(tags_decoded[:]))
        print('{}'.format(intent_decoded))
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('as')
    parser.add_argument('-m', '--model',
                        nargs='*',
                        action='store',
                        dest='models_path',
                        default=config.GDRIVE_TMP_MODELS_PATH,
                        help='where the model is')

    parser.add_argument('-t', '--texts',
                        nargs='*',
                        action='store',
                        dest='texts',
                        default=None,
                        help='text data to run the inference')

    parser.add_argument('-i', '--in',
                        action='store',
                        dest='in_path',
                        default=config.REPO_ROOT,
                        help="where to find the indexes and sets"
                        )
    res = parser.parse_args()
    print(res)

    embedder = gensim.models.KeyedVectors.load_word2vec_format(res.embedder_path, binary=True)

    Variables.POS2OH, \
    Variables.INTENT2OH, \
    Variables.INDEX2INTENT, \
    Variables.TAG2OH, \
    Variables.INDEX2TAG = load_dicts(in_path=res.in_path)

    model = EncoderDecoder()
    model.load_models_to_disk(root=res.models_path)

    test(model, embedder,res.texts)
