import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.layers import LSTM, CuDNNLSTM, CuDNNGRU

from models.word2vec.Loader import Loader
from models.word2vec.w2v import Model
from keras.utils.vis_utils import plot_model


def embed_data(tx, model, vocab):
    tx_temp = []
    for t in tx:
        tmp = []
        for tt in t:
            tmp.append(model.embed(tt, corpus=vocab))
        tx_temp.append(np.array(tmp))
    return np.array(tx_temp)


def encode_y(ys, labels):
    dict = {}
    i = 1
    for label in labels:
        dict[label] = i
        i += 1
    y_int = []
    for y in ys:
        y_int.append(dict[y])
    y_int = np.array(y_int)

    b = np.zeros((y_int.size, y_int.max() + 1))
    b[np.arange(y_int.size), y_int] = 1

    return b


def padding(tx, maxlen=5):
    new = []
    for t in tx:
        if (len(t) < maxlen):
            tmp_t = list(t)
            to_pad = [0 for i in range(len(t[0]))]
            l = len(t)
            while l < maxlen:
                tmp_t.append(to_pad)
                l += 1
            t = np.array(tmp_t)
        new.append(t)
    return np.array(new)


def maxlen(tx):
    return max(len(t) for t in tx)


VECTOR_SIZE = 2

# loader = Loader('/home/weiss/CODES/PFE_M2/nlu/intent_classification/data/traning/AskUbuntuCorpus.json')
# loader = Loader('/home/weiss/CODES/PFE_M2/nlu/intent_classification/data/traning/WebApplicationsCorpus.json')
loader = Loader(
    '/home/weiss/CODES/PFE_M2/nlu/intent_classification/data/traning/ChatbotCorpus.json')
loader.construct_corpus()
loader.normalize_corpus()
loader.save_corpus_to_file()

model = Model()
# model.word2phrase('./corpus.txt',verbose=True)
model.train_model('./corpus.txt', verbose=True, vector_size=VECTOR_SIZE)
print(model.embed('', corpus=loader.words))

# model = Model(path_to_model='./corpus.txt.bin')

LABELS = set(loader.text_dataset[:, -1])
NUM_LABELS = len(LABELS)

tx, ty, tex, tey = loader.split_dataset(0.8, shuffle=True)

tx_embeded = embed_data(tx, model, loader.words)
tex_embeded = embed_data(tex, model, loader.words)

ty_encoded = encode_y(ty, LABELS)
tey_encoded = encode_y(tey, LABELS)

MAX_LEN = maxlen(loader.text_dataset[:, 0])

tx_embeded = padding(tx_embeded, maxlen=MAX_LEN)
tex_embeded = padding(tex_embeded, maxlen=MAX_LEN)

print(tx_embeded.shape)
print(tex_embeded.shape)
print(NUM_LABELS)

encoder_inputs = Input(shape=(None, 15), name='encoder_inputs')
encoder = LSTM(256, return_state=True, name='encoder')
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
encoder_dense = Dense(15, activation='softmax', name='intent_classifier')
intent_output = encoder_dense(encoder_outputs)

decoder_inputs = Input(shape=(None, 15), name='decoder_inputs')
decoder_lstm = LSTM(256, return_state=True,
                    return_sequences=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(15, activation='softmax',
                      name='named_entity_recognition')
decoder_outputs = decoder_dense(decoder_outputs)
model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[
                    intent_output, decoder_outputs], name='general_model')
# model = Sequential()
# model.add(LSTM(50, return_sequences=False,input_shape=(None,VECTOR_SIZE)))
# model.add(Dense(50))
# model.add(Dense(NUM_LABELS+1, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['categorical_accuracy'])
print(model.summary())
plot_model(model, to_file='./model.png',
           show_shapes=True, show_layer_names=True)


# model.fit(tx_embeded,ty_encoded,epochs=10)

# score = model.evaluate(tex_embeded,tey_encoded)
# print("SCORE : ",score[1])
# print('Prediction for {} is {}: '.format(tx_embeded[0],model.predict([[tx_embeded[0]]])))
