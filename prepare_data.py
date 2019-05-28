import argparse
import itertools
import json
import random
import re

import nltk

from config import TrainingConfig as config
from utils import (exclude_condition,
                   fill_placeholders,
                   load_data,
                   make_ids,
                   create_postag_oh_dict,
                   create_intent_oh_dict,
                   create_tag_oh_dict,
                   create_index_to_intent,
                   create_index_to_tag)


class Variables:
    PLUGS = None
    INSTANCES = None
    GROUP = None
    TEST_GROUP = None
    POS2OH = None
    INTENT2OH = None
    INDEX2INTENT = None
    TAG2OH = None
    INDEX2TAG = None


def prepare_plugs(path=None):
    if path is None:
        path = config.PLUGS_PATH
    PLUGS = open(path, 'r').readlines()
    PLUGS = [dir[:-1] for dir in PLUGS if exclude_condition(dir[:-1])]
    PLUGS = [re.sub(r'[-.|_@$ )()]+', '_', dir) for dir in PLUGS]
    PLUGS = [dir[:-1] if dir.endswith('_') else dir for dir in PLUGS]
    PLUGS = [dir.lower() for dir in PLUGS]
    return PLUGS


def fill_dataset(PLUGS, DATASET_PATH=None, DATASET_CLEANED_PATH=None):
    nltk.download('averaged_perceptron_tagger')
    if DATASET_PATH is None:
        DATASET_PATH = config.DATASET_PATH
    if DATASET_CLEANED_PATH is None:
        DATASET_CLEANED_PATH = config.DATASET_CLEANED_PATH
    try:
        f = open(DATASET_CLEANED_PATH, 'w')
        data = load_data(DATASET_PATH)
        print(DATASET_PATH)
        types = [typ for typ in data['train_dataset']]
        for typ in types:
            print(typ)
            data['train_dataset'][typ] = fill_placeholders(data['train_dataset'][typ], PLUGS)
        for typ in types:
            make_ids(data['train_dataset'][typ])
        json.dump(data, f, indent=4, separators=(',', ': '))
    except FileNotFoundError as e:
        print('[ERROR] : ', e)


def load_instances(DATASET_CLEANED_PATH=None):
    if DATASET_CLEANED_PATH is None:
        DATASET_CLEANED_PATH = config.DATASET_CLEANED_PATH

    # I have literally no idea why this shits works
    with open(DATASET_CLEANED_PATH, 'r') as f:
        print(f.readlines()[-10:])

    DATA = load_data(DATASET_CLEANED_PATH)

    config.INTENTS_SET = DATA['intents_set']
    config.TAGS_SET = DATA['tags_set']
    INSTANCES = DATA['train_dataset']
    INSTANCES['others'].extend(1000 * INSTANCES['others'])
    INSTANCES['unknown'].extend(1000 * INSTANCES['unknown'])
    INSTANCES = list(itertools.chain.from_iterable([INSTANCES[t] for t in INSTANCES]))
    random.shuffle(INSTANCES)

    GROUP = {len(inst['postags']): [] for inst in INSTANCES}
    for index, inst in enumerate(INSTANCES):
        GROUP[len(inst['postags'])].append(inst)
    TEST_GROUP = {group: [] for group in list(GROUP)}
    return INSTANCES, GROUP, TEST_GROUP


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

    return POS2OH, INTENT2OH, INDEX2INTENT, TAG2OH, INDEX2TAG


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--plugs',
                        action='store',
                        dest='plugs_path',
                        default=config.PLUGS_PATH,
                        help='Path to the plugs')

    parser.add_argument('-d', '--dims',
                        nargs=2,
                        action='store',
                        dest='dims',
                        default=[64, 64],
                        help=
                        """
                        dims are in this order : 
                            1 : encoder_output_dim
                            2 : encoder_dense_units
                        """
                        )
    parser.add_argument('-b', '--build',
                        action='store',
                        dest='build',
                        default=True,
                        type=bool,
                        help='To build or not the dataset')

    res = parser.parse_args()
    print(res)
    Variables.PLUGS = prepare_plugs(path=res.plugs_path)
    if res.build :
        fill_dataset(Variables.PLUGS)
    Variables.INSTANCES, Variables.GROUP, Variables.TEST_GROUP = load_instances()

    Variables.POS2OH, \
    Variables.INTENT2OH, \
    Variables.INDEX2INTENT, \
    Variables.TAG2OH, \
    Variables.INDEX2TAG = load_dicts(
        model_name=res.model_name,
        encoder_output_dim=res.dims[0],
        encoder_dense_units=res.dims[1])
