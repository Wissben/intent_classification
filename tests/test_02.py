def train_generator(data, w2v_model):
    """
    {
     'id': 2,
     'intent': 'open_file_desire',
     'postags': ['MD', 'PRP', 'VB', 'DT', 'NN', 'NNP', 'NNP', 'NNP', '.'],
     'tags': 'NUL NUL NUL NUL NUL file_name file_name file_name NUL',
     'text': 'Can you open the file AIND VUI Capstone ?'
    }
    """
    stop_cond = False
    index = 0
    while index < len(data):
        instance = data[index]
        x_train, y_train, intents = instance_codifier(instance)
        sh = y_train.shape
        slots = np.zeros(shape=(sh))
        slots[0, 1:, :] = y_train[0, :-1, :]
        yield ({
            "encoder_inputs": x_train,
            "decoder_inputs": y_train
        },
            {
            "intent_classifier": intents,
            "named_entity_recognition": slots
        })
        index += 1
#         stop_cond = True if index >= len(data) else False


def train_batch_generator(data, w2v_model, batch_size=1):
    stop_cond = False
    index = 0
    while not stop_cond:
        batch_x = []
        batch_y = []
        batch_intents = []
        batch_slots = []

        for i in range(batch_size):
            instance = data[index]
            x_train, y_train, intents = instance_codifier(
                instance, batched=True)
            sh = y_train.shape
            slots = np.zeros(shape=(sh))
            slots[1:, :] = y_train[:-1, :]

            batch_x.append(x_train)
            batch_y.append(y_train)
            batch_intents.append(intents)
            batch_slots.append(slots)
            #-----------------------------------#
            # End of batch loop
        # creting np arrays from the collected batches
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_intents = np.array(batch_intents)
        batch_slots = np.array(batch_slots)
        x_train, y_train, slots = pad_sequences(
            x_train, maxlen=conf_obj['max_len']),
        pad_sequences(y_train, maxlen=conf_obj['max_len']), pad_sequences(
            slots, maxlen=conf_obj['max_len']),
        print('Shapes are now {} and {} and {} '.format(
            x_train.shape, y_train.shape, slots.shape))

        # return the batches
        yield ({
            "encoder_inputs": batch_x,
            "decoder_inputs": batch_y
        },
            {
            "intent_classifier": batch_intents,
            "named_entity_recognition": batch_slots
        })
        # reinitializing the batches to empty python lists
        batch_x = []
        batch_y = []
        batch_intents = []
        batch_slots = []
        # checking exist condition
        index += batch_size
        stop_cond = True if index > len(data) else False


def generate_training_set(data, ratio=0.8):
    training_stop_index = int(len(data)*ratio)
    batch_x = []
    batch_y = []
    batch_intents = []
    batch_slots = []
    for i in range(training_stop_index):
        instance = data[index]
        x_train, y_train, intents = instance_codifier(instance, batched=True)
        sh = y_train.shape
        slots = np.zeros(shape=(sh))
        slots[1:, :] = y_train[:-1, :]

        batch_x.append(x_train)
        batch_y.append(y_train)
        batch_intents.append(intents)
        batch_slots.append(slots)
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    batch_intents = np.array(batch_intents)
    batch_slots = np.array(batch_slots)

    return batch_x, batch_y, batch_slots, batch_intents


def instance_codifier(instance, batched=False):
    word_vectors = [w2v(word, w2v_model) for word in instance['text'].split()]
    word_postags_encoded = [POS2OH[w] for w in instance['postags']]
    x_train = np.array([np.concatenate((vec, postag), axis=None)
                        for vec, postag in zip(word_vectors, word_postags_encoded)])
#     x_train = np.array ([ np.array(vec) for vec in word_vectors])
    if(not batched):
        x_train = np.reshape(x_train, newshape=(
            1, x_train.shape[0], x_train.shape[1]))

    y_train = np.array([TAG2OH[t] for t in instance['tags'].split()])
    if(not batched):
        y_train = np.reshape(y_train, newshape=(
            1, y_train.shape[0], y_train.shape[1]))

    intents = np.array([INTENT2OH[instance['intent']]])

    return x_train, y_train, intents


def make_ids(intents):
    for id, intent in enumerate(intents):
        intent['id'] = id+1
    # pr(data)


def fill_placeholders(intents, plugs, separator='_'):
    new = []
    start_id = 1
    for plug in plugs:
        for intent in intents:
            text = intent['text']
            text_tag = re.sub(r'{(.+?):}', r'{\1:<'+plug+'>}', text)
            text = re.sub(r'{(.+?):}', ' '.join(plug.split(separator)), text)
#             text= re.sub(r'{(.+?):}',plug, text)
            postags = nltk.pos_tag(text.split())
            postags = [t for (w, t) in postags]
            tags = get_tags(text_tag.split())
            obj_model = {'id': start_id,
                         'text': text,
                         'tags': tags,
                         'postags': postags,
                         'intent': intent['intent']}
            new.append(obj_model)
            start_id += 1
            if(start_id % 10000 == 0):
                pr(start_id)
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
    res = None
    try:
        res = model[word]
    except Exception as e:
        res = np.zeros(model.vector_size)
    return res
