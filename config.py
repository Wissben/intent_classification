class TrainingConfig:
    REPO_ROOT = '/content/intent_classification/'
    DATASET_CLEANED_PATH = REPO_ROOT+'data/training/slot_filling_datasets/train_intents.json.json'
    DATASET_PATH = REPO_ROOT+'data/training//slot_filling_datasets/train_intents.json'
    GDRIVE_MOUNT_PATH = '/content/gdrive/'
    GDRIVE_PATH = GDRIVE_MOUNT_PATH+'My Drive/'
    GDRIVE_TMP_MODELS_PATH = '/content/gdrive/My Drive/Models_NLU/'
    GDRIVE_MODELS_PATH = '/content/gdrive/My Drive/Models_NLU/best/'

    PLUGS_PATH = REPO_ROOT+'data/traning/slot_filling_datasets/plugs.in'

    INTENTS_SET = list()
    TAGS_SET = list()

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

    CONF_OBJ = {
        'model_name': 'base',
        'encoder_input_dim': 0,
        'encoder_output_dim': 64,
        'encoder_dense_units': 64,
        'encoder_dense_output_dim': 1,
        'decoder_input_dim': 1,
        'decoder_output_dim': 1,
    }

    BATCH_SIZE = 128
    VAL_RATIO = 0.25
