import argparse

import gensim
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model

from config import TrainingConfig as config
from models.seq2seq.EncoderDecoder import EncoderDecoder


class Variables:
    model = None
    callbacks = None
    embedder = None


def prepare_models(lw=None, model_cp=None, embedder_path=None):
    if lw is None:
        lw = {
            "intent_classifier": 1.0,
            "named_entity_recognition": 1.0
        }
    if model_cp is None:
        model_cp = config.GDRIVE_TMP_MODELS_PATH

    if embedder_path is None:
        raise FileNotFoundError

    model = EncoderDecoder(config.CONF_OBJ)
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

    "Backup path : '/content/gdrive/My Drive/GoogleNews-vectors-negative300.bin.gz'"
    embedder = gensim.models.KeyedVectors.load_word2vec_format(embedder_path, binary=True)
    return model, callbacks, embedder


if __name__ == '__main__':
    parser = argparse.ArgumentParser('as')

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
                        help='name of the model')

    parser.add_argument('-e', '--embedder',
                        nargs=1,
                        action='store',
                        dest='embedder_path',
                        default='/content/gdrive/My Drive/GoogleNews-vectors-negative300.bin.gz',
                        help=
                        """
                        Path to the word2vec model
                        """
                        )

    parser.add_argument('-m', '--modelname',
                        action='store',
                        dest='model_name',
                        default="base",
                        help='name of the model')

    res = parser.parse_args()
    print(res)
    config.CONF_OBJ['model_name'] = res.model_name
    Variables.model, Variables.callbacks, Variables.embedder = prepare_models(lw=res.losses_weights,
                                                                              model_cp=res.model_cp,
                                                                              embedder_path=res.embedder_path)
    print(Variables)
