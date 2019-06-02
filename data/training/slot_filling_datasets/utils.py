import os
import json
from pprint import pprint as pr
import re
import pprint


def make_ids(intents):
    for id, intent in enumerate(intents):
        intent['id'] = id+1
    # pr(data)


def fill_placeholders(intents, plugs):
    new = []
    start_id = 1
    for plug in plugs:
        for intent in intents:
            text = intent['text']
            tags = get_tags(text.split())
#             text = re.sub(r'{(.+?):}', r'{\1:'+plug+'}', text)
            text = re.sub(r'{(.+?):}', plug, text)
            obj_model = {'id': start_id,
                         'text': text,
                         'tags': tags,
                         'intent': intent['intent']}
#             pr(obj_model)
            new.append(obj_model)
            start_id += 1
            if(start_id % 1000 == 0):
                pr(start_id)
    return new


def get_tags(words):
    res = []
    for word in words:
        match = re.match(r'{(.+?):}', word)
        if match is not None:
            res.append(match.group(1))
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
        not '|' in dir,
        not len(dir.split('-')) > 3,
        not len(dir) < 3
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


# def get_sub_dirs(path, max_depth=1):
#     folders = []
#     # r=root, d=directories, f = files
#     for r, dirnames, f in os.walk(path):
#         if r.count(os.sep) >= max_depth:
#             del dirnames[:]
#         dirnames = [dir for dir in dirnames if exclude_condition(dir)]
#         for folder in dirnames:
#             folders.append(folder)
#             # print(folder)
#     return folders


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


def change_intents_domains(data, dic):
    i = 0
    for type in data['train_dataset']:
        for inst in data['train_dataset'][type]:
            new_text = re.sub(
                r'{(.+?):}', r'{%s\1:}' % (dic[inst['intent']]), inst['text'])
            inst['text'] = new_text

    # "/home/weiss/Documents/intent_parser/train_intents.json"
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'train_intents.json')
# "/home/weiss/Documents/intent_parser/train_intents_cleaned.json"
DATASET_CLEANED_PATH = os.path.join(
    os.path.dirname(__file__), 'train_intents_cleaned.json')
ROOT_PATH = '/home/weiss'
PLUGS = []

data = load_data(DATASET_PATH)

dic = {'create_file_desire': 'ALTER.',
       'create_directory_desire': 'ALTER.',
       'delete_file_desire': 'ALTER.',
       'delete_directory_desire': 'ALTER.',
       'open_file_desire': 'INFO.',
       'close_file_desire': 'INFO.',
       "copy_file_desire": 'ALTER.',
       "move_file_desire": 'ALTER.',
       "rename_file_desire": 'ALTER.',
       "inform": 'EXCH.',
       "request": "EXCH.",
       "change_directory_desire": '',
       "deny": '',
       "confirm": '',
       "unknown": ''
       }

change_intents_domains(data, dic)
fd = open(DATASET_PATH, 'w')
json.dump(data, fd, sort_keys=True, indent=4, separators=(',', ': '))
