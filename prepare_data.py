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


def save_dicts(out_path, DATASET_CLEANED_PATH=None):
    if DATASET_CLEANED_PATH is None:
        DATASET_CLEANED_PATH = config.DATASET_CLEANED_PATH
    DATA = load_data(DATASET_CLEANED_PATH)

    fd = open(out_path + 'intents_set', 'w')
    json.dump(fd, DATA['intents_set'])
    fd = open(out_path + 'tags_set', 'w')
    json.dump(fd, DATA['tags_set'])


def load_dicts(in_path):
    fd = open(in_path + 'intents_set', 'r')
    INTENTS_SET = json.load(fd)
    fd = open(in_path + 'tags_set', 'r')
    TAGS_SET = json.load(fd)

    POS2OH = create_postag_oh_dict(config.POSTAG_SET)
    INTENT2OH = create_intent_oh_dict(INTENTS_SET)
    TAG2OH = create_tag_oh_dict(TAGS_SET)
    INDEX2INTENT = create_index_to_intent(INTENTS_SET)
    INDEX2TAG = create_index_to_tag(TAGS_SET)

    return POS2OH, INTENT2OH, INDEX2INTENT, TAG2OH, INDEX2TAG


def load_instances(DATASET_CLEANED_PATH=None):
    if DATASET_CLEANED_PATH is None:
        DATASET_CLEANED_PATH = config.DATASET_CLEANED_PATH

    # I have literally no idea why this shits works
    with open(DATASET_CLEANED_PATH, 'r') as f:
        print(f.readlines()[-10:])

    DATA = load_data(DATASET_CLEANED_PATH)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--plugs',
                        action='store',
                        dest='plugs_path',
                        default=config.PLUGS_PATH,
                        help='Path to the plugs')

    parser.add_argument('-b', '--build',
                        action='store',
                        dest='build',
                        default=False,
                        type=bool,
                        help='To build or not the dataset')

    parser.add_argument('-o', '--out',
                        action='store',
                        dest='out_path',
                        default=config.REPO_ROOT,
                        type=str,
                        help='Where to store the indexes')

    res = parser.parse_args()
    print(res)
    Variables.PLUGS = prepare_plugs(path=res.plugs_path)
    if res.build:
        fill_dataset(Variables.PLUGS)
    save_dicts(out_path=res.out_path)
