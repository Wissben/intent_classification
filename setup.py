import itertools
import json
import re
from random import random

import gensim
import nltk
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model

from config import TrainingConfig as config
from models.seq2seq.EncoderDecoder import EncoderDecoder
from utils import (exclude_condition,
                   fill_placeholders,
                   load_data,
                   make_ids,
                   create_postag_oh_dict,
                   create_intent_oh_dict,
                   create_tag_oh_dict,
                   create_index_to_intent,
                   create_index_to_tag,
                   training_loop)

class Variables :
    PLUGS = None
    INSTANCES = None
    GROUP = None
    dicts = None
    models = None

def prepare_plugs():
    PLUGS = open(config.PLUGS_PATH, 'r').readlines()
    PLUGS = [dir[:-1] for dir in PLUGS if exclude_condition(dir[:-1])]
    PLUGS = [re.sub(r'[-.|_@$ )()]+', '_', dir) for dir in PLUGS]
    PLUGS = [dir[:-1] if dir.endswith('_') else dir for dir in PLUGS]
    PLUGS = [dir.lower() for dir in PLUGS]
    return PLUGS


def fill_dataset(PLUGS):
    nltk.download('averaged_perceptron_tagger')
    try:
        f = open(config.DATASET_CLEANED_PATH, 'w')
        data = load_data(config.DATASET_PATH)
        print(config.DATASET_PATH)
        types = [typ for typ in data['train_dataset']]
        for typ in types:
            print(typ)
            data['train_dataset'][typ] = fill_placeholders(data['train_dataset'][typ], PLUGS)
        for typ in types:
            make_ids(data['train_dataset'][typ])
        json.dump(data, f, indent=4, separators=(',', ': '))
    except FileNotFoundError as e:
        print('[ERROR] : ', e)


def load_instances():
    # I have literally no idea why this shits works
    with open(config.DATASET_CLEANED_PATH, 'r') as f:
        print(f.readlines()[-10:])
    DATA = load_data(config.DATASET_CLEANED_PATH)

    config.INTENTS_SET = DATA['intents_set']
    config.TAGS_SET = DATA['tags_set']
    INSTANCES = DATA['train_dataset']
    INSTANCES['others'].extend(50 * INSTANCES['others'])
    INSTANCES['unknown'].extend(50 * INSTANCES['unknown'])
    INSTANCES = list(itertools.chain.from_iterable([INSTANCES[t] for t in INSTANCES]))
    random.shuffle(INSTANCES)

    GROUP = {len(inst['postags']): [] for inst in INSTANCES}
    for index, inst in enumerate(INSTANCES):
        GROUP[len(inst['postags'])].append(inst)

    return INSTANCES, GROUP


def load_dicts(**kwargs):
    POS2OH = create_postag_oh_dict(config.POSTAG_SET)
    INTENT2OH = create_intent_oh_dict(config.INTENTS_SET)
    TAG2OH = create_tag_oh_dict(config.TAGS_SET)
    INDEX2INTENT = create_index_to_intent(config.INTENTS_SET)
    INDEX2TAG = create_index_to_tag(config.TAGS_SET)

    n_intents = len(INTENT2OH)
    n_tags = len(TAG2OH)
    vector_size = 300  # w2v_model.vector_size
    postag_vec_size = len(config.POSTAG_SET)

    config.CONF_OBJ = {
        'model_name': kwargs['model_name'] if 'model_name' in list(kwargs) else 'base',
        'encoder_input_dim': vector_size + postag_vec_size,
        'encoder_output_dim': kwargs['encoder_output_dim'] if 'encoder_output_dim' in list(kwargs) else 64,
        'encoder_dense_units': kwargs['encoder_dense_units'] if 'encoder_dense_units' in list(kwargs) else 64,
        'encoder_dense_output_dim': n_intents,
        'decoder_input_dim': n_tags,
        'decoder_output_dim': n_tags,
    }

    return {
        'POS2OH': POS2OH,
        'INTENT2OH': INTENT2OH,
        'INDEX2INTENT': INDEX2INTENT,
        'TAG2OH': TAG2OH,
        'INDEX2TAG': INDEX2TAG
    }


def prepare_models():
    model = EncoderDecoder(config.CONF_OBJ)
    model.save_model_image()
    plot_model(model.inf_decoder_model,
               to_file='./{}.png'.format('inf_decoder_model'),
               show_shapes=True,
               show_layer_names=True)
    plot_model(model.inf_encoder_model,
               to_file='./{}.png'.format('inf_encoder_model'),
               show_shapes=True,
               show_layer_names=True)
    plot_model(model.inf_intent_classifier,
               to_file='./{}.png'.format('inf_intent_classifier'),
               show_shapes=True,
               show_layer_names=True)

    model.model.summary()

    losses = {
        "intent_classifier": "categorical_crossentropy",
        "named_entity_recognition": "categorical_crossentropy",
    }

    metrics = {
        "intent_classifier": "categorical_accuracy",
        "named_entity_recognition": "categorical_accuracy",
    }

    lossWeights = {
        "intent_classifier": 1.0,
        "named_entity_recognition": 1.0
    }

    model.model.compile(optimizer=RMSprop(),
                        loss=losses,
                        loss_weights=lossWeights,
                        metrics=metrics)

    callbacks = [
        ModelCheckpoint(filepath=config.GDRIVE_TMP_MODELS_PATH + '{}'.format(model.model_name),
                        monitor='val_loss',
                        verbose=0,
                        save_best_only=False,
                        save_weights_only=False,
                        mode='auto',
                        period=1)
    ]

    embbeder = gensim.models.KeyedVectors.load_word2vec_format('gdrive/My Drive/GoogleNews-vectors-negative300.bin.gz',
                                                               binary=True)

    return {'model': model, 'callbacks': callbacks, 'embedder': embbeder}


if __name__ == '__main__':

    Variables.PLUGS = prepare_plugs()
    print(Variables.PLUGS)
    # fill_dataset(Variables.PLUGS)
    Variables.INSTANCES, Variables.GROUP = load_instances()
    Variables.dicts = load_dicts(model_name='test',encoder_output_dim=256)
    Variables.models = prepare_models()
