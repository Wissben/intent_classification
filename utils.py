import json
import os
import re
from pprint import pprint as pr
import random

import nltk
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import TrainingConfig as config


def create_slotless_entries(name):
    with open(name + '.in', 'r') as fd:
        lines = [line[:-1].lower() for line in fd.readlines()]
        instances = [{
            "id": 1,
            "text": line,
            "intent": "deny"
        } for line in lines]
        for instance in instances:
            if 'ok' in instance:
                instances.append({
                    "id": 1,
                    "text": "it is not correct",
                    "intent": "deny"
                })
        return instances

def train_batch_generator(data, embedder, steps, batch_size=1, padded=False, sample_weights=None,dicts=None):
    random.shuffle(data)
    index = 1
    while True:
        batch_x, batch_y, batch_intents, batch_y_input = fetch_batch_from_data(index, data, batch_size,embedder=embedder,dicts=dicts)
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


def val_batch_generator(data, embedder, steps, batch_size=1, padded=False, sample_weights=None,dicts=None):
    random.shuffle(data)
    index = 1
    while True:
        batch_x, batch_y, batch_intents, batch_y_input = fetch_batch_from_data(index, data, batch_size,embedder=embedder,dicts=dicts)
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


# def train_on_batch_gen(group, w2v_model, batch_size=1, padded=True):
#     for i, gr in enumerate(group):
#         data = group[gr]
#         random.shuffle(data)
#         range_index = [i for i in range(int(len(data) / batch_size) + 1)]
#         for index in range_index:  # range(0,int(len(data)/batch_size)):
#             batch_x, batch_y, batch_intents, batch_y_input = fetch_batch_from_data(index + 1, data, batch_size)
#             batch_intents = np.reshape(batch_intents, (batch_intents.shape[0], batch_intents.shape[2]))
#             # Padding the sequences
#             if padded:
#                 batch_x = pad_sequences(batch_x, maxlen=config.CONF_OBJ['max_len'], value=-1)
#                 batch_y = pad_sequences(batch_y, batch_y_input=config.CONF_OBJ['max_len'])
#                 batch_y_input = pad_sequences(batch_y_input, maxlen=config.CONF_OBJ['max_len'])
#             # The batches are ready to be yielded
#             inputs = {
#                 "encoder_inputs": batch_x,
#                 "decoder_inputs": batch_y_input
#             }
#             outputs = {
#                 "intent_classifier": batch_intents,
#                 "named_entity_recognition": batch_y
#             }
#
#             yield (inputs, outputs)
#         print('[INFO] : Done training for group N° {} with lengths : {} {}'.format(i + 1,
#                                                                                    [len(input) for input in inputs],
#                                                                                    [len(output) for output in outputs]))


def fetch_batch_from_data(batch_index, data, batch_size,embedder=None,dicts=None):
    base = (batch_index - 1) * batch_size
    offset = base + batch_size
    if (offset) > len(data):
        tmp = data[base:]
        #       padding = [tmp[0] for i in range(offset-len(data))]
        #       pr('padding {}'.format(len(padding)))
        print('\ndata {} base {} offset {} '.format(len(data), base, offset))
    #       tmp.extend(padding)
    #       pr('tmp {}'.format(len(tmp)))
    else:
        tmp = data[base:offset]

    batch_x = []
    batch_y = []
    batch_intents = []
    batch_y_input = []

    for i in range(len(tmp)):
        instance = tmp[i]
        x_train, y_train, intents = instance_codifier(instance,embedder=embedder,batched=True,dicts=dicts)

        y_input = np.zeros(shape=y_train.shape)
        y_input[1:, :] = y_train[:-1, :]
        batch_x.append(x_train)
        batch_y.append(y_train)
        batch_intents.append(intents)
        batch_y_input.append(y_input)
        # -----------------------------------#
        # End of batch loop

    # creting np arrays from the collected batches
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    batch_intents = np.array(batch_intents)
    batch_y_input = np.array(batch_y_input)

    return batch_x, batch_y, batch_intents, batch_y_input


def instance_codifier(instance,embedder,batched=False,dicts=None):
    word_vectors = [w2v(word, embedder) for word in instance['text'].split()]
    word_postags_encoded = [dicts['POS2OH'][w] for w in instance['postags']]

    x_train = np.array(
        [np.concatenate((vec, postag), axis=None) for vec, postag in zip(word_vectors, word_postags_encoded)])
    if (not batched):
        x_train = np.reshape(x_train, newshape=(1, x_train.shape[0], x_train.shape[1]))

    y_train = np.array([dicts['TAG2OH'][t] for t in instance['tags'].split()])
    if (not batched):
        y_train = np.reshape(y_train, newshape=(1, y_train.shape[0], y_train.shape[1]))

    intents = np.array([dicts['INTENT2OH'][instance['intent']]])

    return x_train, y_train, intents


def make_ids(intents):
    for id, intent in enumerate(intents):
        intent['id'] = id + 1
    # pr(data)


def fill_placeholders(instances, plugs, separator='_'):
    new = []
    start_id = 1
    # We loop through all the data
    for instance in instances:
        # Intent doesn't have arguments
        if instance['intent'] in ['deny', 'confirm', 'unknown']:
            text = instance['text'].lower()
            # we postag the text
            postags = nltk.pos_tag(text.split())
            postags = [t for (w, t) in postags]
            # We fill the tags with NUL
            tags = ' '.join(['NUL' for word in text.split()])
            obj_model = {'id': start_id,
                         'text': text,
                         'tags': tags,
                         'postags': postags,
                         'intent': instance['intent']}
            new.append(obj_model)
            # Debuging print
            # --------------------------------
            start_id += 1
            pr(start_id)
            # --------------------------------
        else:
            for plug in plugs:
                o1 = random.choice(plugs)
                o2 = random.choice(plugs)
                o3 = random.choice(plugs)
                o4 = random.choice(plugs)
                o5 = random.choice(plugs)
                o6 = random.choice(plugs)

                text = instance['text'].lower()

                # A bit harcoded here, could loop through all the tags and pic randomly another plug
                text_tag = re.sub(r'{(file_name):}', r'{\1:<' + '_'.join(plug.split(separator)) + '>}', text)
                text_tag = re.sub(r'{(parent_directory):}', r'{\1:<' + '_'.join(o1.split(separator)) + '>}', text_tag)
                text_tag = re.sub(r'{(new_directory):}', r'{\1:<' + '_'.join(o2.split(separator)) + '>}', text_tag)
                text_tag = re.sub(r'{(directory_name):}', r'{\1:<' + '_'.join(o3.split(separator)) + '>}', text_tag)
                text_tag = re.sub(r'{(origin):}', r'{\1:<' + '_'.join(o3.split(separator)) + '>}', text_tag)
                text_tag = re.sub(r'{(dest):}', r'{\1:<' + '_'.join(o4.split(separator)) + '>}', text_tag)
                text_tag = re.sub(r'{(old_name):}', r'{\1:<' + '_'.join(o5.split(separator)) + '>}', text_tag)
                text_tag = re.sub(r'{(new_name):}', r'{\1:<' + '_'.join(o6.split(separator)) + '>}', text_tag)

                new_text = re.sub(r'{(.+?):<(.+?)>}', lambda match: ' '.join(match.group(2).split(separator)), text_tag)

                postags = nltk.pos_tag(new_text.split())
                postags = [t for (w, t) in postags]
                tags = get_tags(text_tag.split())
                obj_model = {'id': 1,
                             'text': new_text,
                             'tags': tags,
                             'postags': postags,
                             'intent': instance['intent']}
                new.append(obj_model)
                # Debuging print
                # --------------------------------
                start_id += 1
                if (start_id % 10000 == 0):
                    pr(start_id)
                # --------------------------------
    return new


def get_tags(words, separator='_'):
    res = []
    for word in words:
        match = re.match(r'{(.+?):(<.+?>)}', word)
        if match is not None:
            spl = match.group(2)[1:-1].split(separator)
            res.extend([match.group(1) for sub in spl])
        else:
            res.append('NUL')
    return ' '.join(res)


def exclude_condition(dir):
    conds = [
        dir not in ['lib', 'bin', 'logs', 'log'],
        not dir.startswith('.'),
        not dir.startswith('_'),
        not dir.startswith('-'),
        not dir.startswith('@'),
        not re.match(r'^[0-9 .-_|@]+$', dir),
        not re.match(r'[0-9]+-[0-9]+-[0-9]+', dir),
        not re.match(r'[0-9]+-[0-9]+', dir),
        not re.match(r'^[0-9]+$', dir),
        not re.match(r'^[0-9a-fA-Fa-z-]+$', dir),
        not re.search(r'[0-9]+', dir),
        not '|' in dir,
        not len(re.compile(r'[-.|_]').split(dir)) > 3,
        not len(dir) < 3,
        not dir is ''
    ]

    return all(conds)


def listdir(path):
    """
    recursively walk directory to specified depth
    :param path: (str) path to list files from
    :yields: (str) filename, including path
    """
    for filename in os.listdir(path):
        yield os.path.join(path, filename)


def walk(path='.', depth=None):
    """
    recursively walk directory to specified depth
    :param path: (str) the base path to start walking from
    :param depth: (None or int) max. recursive depth, None = no limit
    :yields: (str) filename, including path
    """
    if depth and depth == 1:
        for filename in listdir(path):
            yield filename
    else:
        top_pathlen = len(path) + len(os.path.sep)
        for dirpath, dirnames, filenames in os.walk(path):
            dirlevel = dirpath[top_pathlen:].count(os.path.sep)
            if depth and dirlevel >= depth:
                dirnames[:] = []
            else:
                for filename in dirnames:
                    yield filename


def load_data(path):
    try:
        fd = open(path, 'r')
        data = json.load(fd)
        return data
    except Exception as e:
        print(e)
        return None


def n_words(data):
    types = [typ for typ in data['train_dataset']]
    res = set()
    for typ in types:
        for instance in data['train_dataset'][typ]:
            text = re.sub(r'{(.+?):}', '', instance['text'])
            res = res | set(text.split())
    return res


def create_postag_oh_dict(postag_set):
    n_features = len(postag_set)
    res = {tag: [1 if i == index else 0 for i in range(
        n_features)] for index, tag in enumerate(postag_set)}
    return res


def create_intent_oh_dict(intents_set):
    n_features = len(intents_set)
    res = {intent: [1 if i == index else 0 for i in range(
        n_features)] for index, intent in enumerate(intents_set)}
    return res


def create_tag_oh_dict(tags_set):
    n_features = len(tags_set)
    res = {intent: [1 if i == index else 0 for i in range(
        n_features)] for index, intent in enumerate(tags_set)}
    return res


def create_index_to_intent(intents_set):
    res = {index: tag for index, tag in enumerate(intents_set)}
    return res


def create_index_to_tag(tags_set):
    res = {index: tag for index, tag in enumerate(tags_set)}
    return res


def w2v(word, model):
    res = np.zeros(model.vector_size)
    if (word == '\start'):
        res[0] = 1
        return res
    if (word == '\end'):
        res[-1] = 1
        return res
    try:
        res = model[word]
    except Exception as e:
        res = model['UNK']
    return res


def training_loop(models,GROUP):
    for group in GROUP:
        data = GROUP[group]
        #   random.shuffle(data)

        val_index = int(len(data) * config.VAL_RATIO + 1)
        data_train = data[:val_index]
        data_val = data[val_index:]
        # Should change this to skip validation data if not enough samples
        #   if(len(data_val)) == 0:
        #     data_vale=data_train
        steps_per_epoch = int(len(data_train) / config.BATCH_SIZE + 1)
        validation_steps = int(len(data_val) / config.BATCH_SIZE + 1)

        hist = models['model'].model.fit_generator(
            generator=train_batch_generator(data_train, models['embedder'], steps_per_epoch, config.BATCH_SIZE),
            validation_data=val_batch_generator(data_val, models['embedder'], validation_steps, config.BATCH_SIZE),
            epochs=1,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            workers=1,
            callbacks=models['callbacks']
            #                               class_weight=class_weights
            #                               steps_per_epoch = 200
            )
    return models,hist