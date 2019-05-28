import argparse

from models.seq2seq.generators import train_batch_generator, val_batch_generator
from prepare_data import Variables as v
from config import TrainingConfig as config
from prepare_model import Variables as vt


def train(test_ratio=None, val_ratio=None, batch_size=None, epochs=None):
    if test_ratio is None:
        test_ratio = config.TEST_RATIO
    if val_ratio is None:
        val_ratio = config.VAL_RATIO
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if epochs is None:
        epochs = 1
    for group in v.GROUP:
        print('CURRENT GROUP IS {}'.format(group))

        data = v.GROUP[group]
        test_index = int(len(data) * test_ratio + 1)
        data_train = data[:-test_index]
        v.TEST_GROUP[group] += data[-test_index:]

        val_index = int(len(data_train) * val_ratio + 1)
        data_val = data_train[-val_index:]

        train_steps_bound = 1 if len(data_train) % batch_size != 0 else 0
        val_steps_bound = 1 if len(data_val) % batch_size != 0 else 0

        steps_per_epoch = int(len(data_train) / batch_size + train_steps_bound)
        validation_steps = int(len(data_val) / batch_size + val_steps_bound)

        train_H = vt.model.model.fit_generator(
            generator=train_batch_generator(data_train, vt.embedder, steps_per_epoch, batch_size),
            validation_data=val_batch_generator(data_val, vt.embedder, validation_steps, batch_size),
            epochs=epochs,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            workers=1,
            callbacks=vt.callbacks
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

    res = parser.parse_args()
    print(res)
    train(test_ratio=res.test_ratio,
          val_ratio=res.val_ratio,
          batch_size=res.batch_size,
          epochs=res.batch_size)
