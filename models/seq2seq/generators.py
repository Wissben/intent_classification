import random

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import TrainingConfig as config
from utils import instance_codifier, fetch_batch_from_data


def test_generator(data, embedder, steps=250, sample_weights=None):
    index = 1
    while True:
        instance = data[index]
        x_train, y_train, intents = instance_codifier(instance, embedder=embedder)
        y_input = np.zeros(shape=y_train.shape)
        y_input[0, 1:, :] = y_train[0, :-1, :]
        if index == steps:
            index = 1
        else:
            index += 1

        inputs = [x_train, y_input]
        outputs = [intents, y_train]

        if sample_weights:
            yield (inputs, outputs, sample_weights)
        else:
            yield (inputs, outputs)


def train_batch_generator(data, embedder, steps, batch_size=1, padded=False, sample_weights=None, dicts=None):
    """
    Generator that yields batches of data of same length from an object like instance
    :param data: the instances of data
    :type data: dict
    :param embedder: word2vec model used for word embedding
    :type embedder: Word2VecKeyedVectors
    :param steps: number of steps per epochs
    :type steps: int
    :param batch_size: the size of one batch of data
    :type batch_size: int
    :param padded: to pad or not the data
    :type padded; bool
    :param sample_weights: an array like object of weights assosciated to each sample
    :type sample_weights: list
    :param dicts: set of data indexing dictionaries
    :param dicts: dict
    :return: a tuple of inputs,outputs or inputs,outputs,sample_weights if sample_weight=True
    """

    random.shuffle(data)
    index = 1
    while True:
        batch_x, batch_y, batch_intents, batch_y_input = fetch_batch_from_data(index,
                                                                               data,
                                                                               batch_size,
                                                                               embedder=embedder,
                                                                               dicts=dicts)
        batch_intents = np.reshape(batch_intents, (batch_intents.shape[0], batch_intents.shape[2]))

        # Padding the sequences
        if padded:
            batch_x = pad_sequences(batch_x, maxlen=config.CONF_OBJ['max_len'], value=-1)
            batch_y = pad_sequences(batch_y, maxlen=config.CONF_OBJ['max_len'])
            batch_intents = np.reshape(batch_intents, (batch_intents.shape[0], batch_intents.shape[2]))
            batch_y_input = pad_sequences(batch_y_input, maxlen=config.CONF_OBJ['max_len'])

        if index == steps:
            index = 1
        else:
            index += 1

        inputs = [batch_x, batch_y_input]
        outputs = [batch_intents, batch_y]

        if sample_weights:
            yield (inputs, outputs, sample_weights)
        else:
            yield (inputs, outputs)


def val_batch_generator(data, embedder, steps, batch_size=1, padded=False, sample_weights=None, dicts=None):
    """
    Generator that yields batches of data of same length from an object like instance
    :param data: the instances of data
    :type data: dict
    :param embedder: word2vec model used for word embedding
    :type embedder: Word2VecKeyedVectors
    :param steps: number of steps per epochs
    :type steps: int
    :param batch_size: the size of one batch of data
    :type batch_size: int
    :param padded: to pad or not the data
    :type padded; bool
    :param sample_weights: an array like object of weights assosciated to each sample
    :type sample_weights: list
    :param dicts: set of data indexing dictionaries
    :param dicts: dict
    :return: a tuple of inputs,outputs or inputs,outputs,sample_weights if sample_weight=True
    """

    random.shuffle(data)
    index = 1
    while True:
        batch_x, batch_y, batch_intents, batch_y_input = fetch_batch_from_data(index,
                                                                               data,
                                                                               batch_size,
                                                                               embedder=embedder,
                                                                               dicts=dicts)
        batch_intents = np.reshape(batch_intents, (batch_intents.shape[0], batch_intents.shape[2]))

        # Padding the sequences
        if padded:
            batch_x = pad_sequences(batch_x, maxlen=config.CONF_OBJ['max_len'], value=-1)
            batch_y = pad_sequences(batch_y, maxlen=config.CONF_OBJ['max_len'])
            batch_intents = np.reshape(batch_intents, (batch_intents.shape[0], batch_intents.shape[2]))
            batch_y_input = pad_sequences(batch_y_input, maxlen=config.CONF_OBJ['max_len'])

        if index == steps:
            index = 1

        else:
            index += 1

        inputs = [batch_x, batch_y_input]
        outputs = [batch_intents, batch_y]

        if sample_weights:
            yield (inputs, outputs, sample_weights)
        else:
            yield (inputs, outputs)


def test_batch_generator(data, embedder, steps, batch_size=1, sample_weights=None, dicts=None):
    """
    Generator that yields batches of data of same length from an object like instance
    :param data: the instances of data
    :type data: dict
    :param embedder: word2vec model used for word embedding
    :type embedder: Word2VecKeyedVectors
    :param steps: number of steps per epochs
    :type steps: int
    :param batch_size: the size of one batch of data
    :type batch_size: int
    :param sample_weights: an array like object of weights assosciated to each sample
    :type sample_weights: list
    :param dicts: set of data indexing dictionaries
    :param dicts: dict
    :return: a tuple of inputs,outputs or inputs,outputs,sample_weights if sample_weight=True
    """

    random.shuffle(data)
    index = 1
    while True:
        batch_x, batch_y, batch_intents, batch_y_input = fetch_batch_from_data(index,
                                                                               data,
                                                                               batch_size,
                                                                               embedder=embedder,
                                                                               dicts=dicts)
        batch_intents = np.reshape(batch_intents, (batch_intents.shape[0], batch_intents.shape[2]))
        if index == steps:
            index = 1

        else:
            index += 1

        inputs = [batch_x, batch_y_input]
        outputs = [batch_intents, batch_y]

        if sample_weights:
            yield (inputs, outputs, sample_weights)
        else:
            yield (inputs, outputs)
