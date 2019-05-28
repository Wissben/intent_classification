import argparse

import gensim
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model

from config import TrainingConfig as config
from models.seq2seq.EncoderDecoder import EncoderDecoder
from models.seq2seq.generators import train_batch_generator, val_batch_generator
from prepare_data import load_dicts
from prepare_data import load_instances
from prepare_model import Variables as vt


class Variables:
    INSTANCES = None
    GROUP = None
    TEST_GROUP = None


def prepare_models(lw=None, model_cp=None, conf_obj=None):
    if conf_obj is None:
        raise AttributeError
    if lw is None:
        lw = {
            "intent_classifier": 1.0,
            "named_entity_recognition": 1.0
        }
    if model_cp is None:
        model_cp = config.GDRIVE_TMP_MODELS_PATH

    model = EncoderDecoder(**conf_obj)
    model.save_model_image()
    plot_model(model.inf_decoder_model,
               to_file='./{}_{}.png'.format(model.model_name, 'inf_decoder_model'),
               show_shapes=True,
               show_layer_names=True)
    plot_model(model.inf_encoder_model,
               to_file='./{}_{}.png'.format(model.model_name, 'inf_encoder_model'),
               show_shapes=True,
               show_layer_names=True)
    plot_model(model.inf_intent_classifier,
               to_file='./{}_{}.png'.format(model.model_name, 'inf_intent_classifier'),
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

    model.model.compile(optimizer=RMSprop(),
                        loss=losses,
                        loss_weights=lw,
                        metrics=metrics)

    callbacks = [
        ModelCheckpoint(filepath=model_cp + '{}'.format(model.model_name),
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=False,
                        save_weights_only=False,
                        mode='auto',
                        period=1)
    ]

    return model, callbacks


def train(model,callbacks,test_ratio=None, val_ratio=None, batch_size=None, epochs=None, embedder=None):
    if test_ratio is None:
        test_ratio = config.TEST_RATIO
    if val_ratio is None:
        val_ratio = config.VAL_RATIO
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if epochs is None:
        epochs = 1
    if embedder is None :
        raise AttributeError

    for group in Variables.GROUP:
        print('CURRENT GROUP IS {}'.format(group))

        data = Variables.GROUP[group]
        test_index = int(len(data) * test_ratio + 1)
        data_train = data[:-test_index]
        Variables.TEST_GROUP[group] += data[-test_index:]

        val_index = int(len(data_train) * val_ratio + 1)
        data_val = data_train[-val_index:]

        train_steps_bound = 1 if len(data_train) % batch_size != 0 else 0
        val_steps_bound = 1 if len(data_val) % batch_size != 0 else 0

        steps_per_epoch = int(len(data_train) / batch_size + train_steps_bound)
        validation_steps = int(len(data_val) / batch_size + val_steps_bound)

        train_H = model.model.fit_generator(
            generator=train_batch_generator(data_train, embedder, steps_per_epoch, batch_size),
            validation_data=val_batch_generator(data_val, embedder, validation_steps, batch_size),
            epochs=epochs,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            workers=1,
            callbacks=callbacks
        )

    return train_H


if __name__ == '__main__':
    parser = argparse.ArgumentParser('as')
    parser.add_argument('-tr', '--testratio',
                        action='store',
                        dest='test_ratio',
                        default=None,
                        type=float,
                        help='Indicates splitting ratio for the testing data')
    parser.add_argument('-vr', '--valratio',
                        action='store',
                        dest='val_ratio',
                        default=None,
                        type=float,
                        help='Indicates splitting ratio for the validation data')
    parser.add_argument('-bs', '--batchsize',
                        action='store',
                        dest='batch_size',
                        default=None,
                        type=int,
                        help='Indicates the size of one batch of data')
    parser.add_argument('-e', '--epochs',
                        action='store',
                        dest='epochs',
                        default=None,
                        type=int,
                        help='Indicates the number of epochs')

    parser.add_argument('-t', '--trainpath',
                        action='store',
                        dest='train_path',
                        default=config.DATASET_CLEANED_PATH,
                        type=str,
                        help='Indicates where the dataset resides')

    parser.add_argument('-e', '--embedder',
                        nargs=1,
                        action='store',
                        dest='embedder_path',
                        default='/content/gdrive/My Drive/GoogleNews-vectors-negative300.bin.gz',
                        help=
                        """
                        Path to the word2vec model
                        """)

    parser.add_argument('-w', '--lw',
                        nargs=2,
                        action='store',
                        dest='losses_weights',
                        default=[1.0, 1.0],
                        type=float,
                        help='Losses weight to apply in the training')

    parser.add_argument('-mcp', '--modelcp',
                        action='store',
                        dest='model_cp',
                        type=str,
                        default=None,
                        help='Checkpoint out path')

    parser.add_argument('-m', '--modelname',
                        action='store',
                        dest='model_name',
                        default="base",
                        help='name of the model')
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
                        """)
    parser.add_argument('-i', '--in',
                        action='store',
                        dest='in_path',
                        default=config.REPO_ROOT,
                        help="where to find the indexes and sets"
                        )

    res = parser.parse_args()
    print(res)

    embedder = gensim.models.KeyedVectors.load_word2vec_format(res.embedder_path, binary=True)
    POS2OH, INTENT2OH, INDEX2INTENT, TAG2OH, INDEX2TAG = load_dicts(in_path=res.in_path)

    n_intents = len(INTENT2OH)
    n_tags = len(TAG2OH)
    vector_size = embedder.vector_size  # w2v_model.vector_size
    postag_vec_size = len(config.POSTAG_SET)

    CONF_OBJ = {
        'model_name': res.model_name,
        'encoder_input_dim': vector_size + postag_vec_size,
        'encoder_output_dim': res.dims[0],
        'encoder_dense_units': res.dims[1],
        'encoder_dense_output_dim': n_intents,
        'decoder_input_dim': n_tags,
        'decoder_output_dim': n_tags,
    }

    model, callbacks = prepare_models(lw=res.losses_weights,
                                      model_cp=res.model_cp,
                                      conf_obj=CONF_OBJ)

    Variables.INSTANCES, Variables.GROUP, Variables.TEST_GROUP = load_instances(DATASET_CLEANED_PATH=res.train_path)

    train(model=model,
          callbacks=callbacks,
          test_ratio=res.test_ratio,
          val_ratio=res.val_ratio,
          batch_size=res.batch_size,
          epochs=res.batch_size)
