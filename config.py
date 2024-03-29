import os


class TrainingConfig:
    REPO_ROOT = os.path.dirname(__file__)
    DATASET_CLEANED_PATH = os.path.join('train_cleaned_intents.json')
    DATASET_PATH = os.path.join(REPO_ROOT, 'data/training/slot_filling_datasets/train_intents.json')
    GDRIVE_MOUNT_PATH = '/content/gdrive/'
    GDRIVE_PATH = GDRIVE_MOUNT_PATH + 'My Drive/'
    GDRIVE_TMP_MODELS_PATH = '/content/gdrive/My Drive/Models_NLU/'
    GDRIVE_MODELS_PATH = '/content/gdrive/My Drive/Models_NLU/best/'

    PLUGS_PATH = os.path.join(REPO_ROOT, 'data/training/slot_filling_datasets/plugs.in')

    POSTAG_SET = {
        "''": 'Unknown',
        "$": 'Unknown',
        "(": 'Unknown',
        ")": 'Unknown',
        ":": 'Unknown',
        'PAD': 'padding tag for training in many batches',
        '.': 'Punctuation',
        'CC': 'coordinating conjunction',
        'CD': 'cardinal digit',
        'DT': 'determiner',
        'EX': 'existential there (like: \"there is\" ... think of it like \"there exists\")',
        'FW': 'foreign word',
        'IN': 'preposition/subordinating conjunction',
        'JJ': 'adjective \'big\'',
        'JJR': 'adjective, comparative \'bigger\'',
        'JJS': 'adjective, superlative \'biggest\'',
        'LS': 'list marker 1)',
        'MD': 'modal could, will',
        'NN': 'noun, singular \'desk\'',
        'NNS': 'noun plural \'desks\'',
        'NNP': 'proper noun, singular \'Harrison\'',
        'NNPS': 'proper noun, plural \'Americans\'',
        'PDT': 'predeterminer \'all the kids\'',
        'POS': 'possessive ending parent\'s',
        'PRP': 'personal pronoun I, he, she',
        'PRP$': 'possessive pronoun my, his, hers',
        'RB': 'adverb very, silently,',
        'RBR': 'adverb, comparative better',
        'RBS': 'adverb, superlative best',
        'RP': 'particle give up',
        'TO': 'to go \'to\' the store.',
        'UH': 'interjection errrrrrrrm',
        'VB': 'verb, base form take',
        'VBD': 'verb, past tense took',
        'VBG': 'verb, gerund/present participle taking',
        'VBN': 'verb, past participle taken',
        'VBP': 'verb, sing. present, non-3d take',
        'VBZ': 'verb, 3rd person sing. present takes',
        'WDT': 'wh-determiner which',
        'WP': 'wh-pronoun who, what',
        'WP$': 'possessive wh-pronoun whose',
        'WRB': 'wh-abverb where, when'
    }

    BATCH_SIZE = 128
    VAL_RATIO = 0.15
    TEST_RATIO = 0.25

    TEST_TEXTS = [
        'perfect do that',
        'could you just do it man',
        'could you move aaa to baaadd',
        'yes please',
        'show me the home directory',
        'go to the home directory',
        'show the file named code.py',
        'delete home diretory',
        'go to the java project folder',
        'sounds good',
        'what the weather like today',
        'rename the whole to dd',
        'Yeah just do that',
        'hey open the home folder',
        'change name of old to new',
        'close the current folder',
        'erase the current folder',
        'Remove the current folder'
    ]
