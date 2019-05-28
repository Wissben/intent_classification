import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (Bidirectional,
                                     CuDNNLSTM,
                                     Dense,
                                     Input,
                                     TimeDistributed,
                                     Concatenate)

from tensorflow.keras.models import load_model as lm
from tensorflow.keras.utils import plot_model


class EncoderDecoder:
    """
    Class implementation for the encoder decoder architecture
    """

    def __init__(self, **kwargs):
        """
        Must provide a dictionary containing the following keys/values :
        {_
            'model_name': 'base',
            'encoder_input_dim': vector_size+postag_vec_size,
            'encoder_output_dim': 64,
            'encoder_dense_units': 64,
            'encoder_dense_output_dim' :n_intents,
            'decoder_input_dim': n_tags,
            'decoder_output_dim': n_tags,
        }
        """
        for arg in kwargs:
            self.__setattr__(arg, kwargs[arg])
        self.build_model()

    def build_model(self):
        """
        Helper function to build the layers of the model
        """
        # Define training encoder
        encoder_inputs = Input(shape=(None, self.encoder_input_dim),
                               name='encoder_inputs')
        # Creating the LSTM cell
        encoder_lstm = CuDNNLSTM(self.encoder_output_dim,
                                 return_state=True,
                                 return_sequences=False,
                                 name='encoder_lstm')
        # Wrapping it into a bidirectionnal layer
        encoder_bilstm = Bidirectional(encoder_lstm, merge_mode='concat')
        # The bidirectional wraper returns 5 variables
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(
            encoder_inputs)
        # We concatenate the left/right context of both
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        #         encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        # And save the hidden and internal states
        encoder_states = [state_h, state_c]
        # Define the dense output layer for the intent classification
        x = Dense(self.encoder_dense_units,
                  activation='relu',
                  name='encoder_hidden_dense_1')(encoder_outputs)

        encoder_dense = Dense(self.encoder_dense_output_dim,
                              activation='softmax',
                              name='intent_classifier')
        # This layer will classify the intent
        intent_output = encoder_dense(x)

        returned = [encoder_inputs, intent_output, encoder_states]

        # Define training decoder
        decoder_inputs = Input(shape=(None, self.decoder_input_dim),
                               name='decoder_inputs')

        decoder_lstm = CuDNNLSTM(2 * self.encoder_output_dim,
                                 return_state=True,
                                 return_sequences=True,
                                 name='decoder_lstm')

        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.decoder_output_dim,
                              activation='softmax',
                              name='named_entity_recognition')

        decoder_outputs = TimeDistributed(decoder_dense, name='named_entity_recognition')(decoder_outputs)
        #       decoder_outputs = CRF(self.decoder_output_dim)(decoder_outputs)
        #       decoder_outputs = decoder_dense(decoder_outputs)
        returned = [decoder_inputs, decoder_outputs]
        # The traning model
        self.model = keras.Model(inputs=[encoder_inputs, decoder_inputs],
                                 outputs=[intent_output, decoder_outputs],
                                 name=self.model_name)

        # define inference_intent_classifier
        self.inf_intent_classifier = keras.Model(encoder_inputs, intent_output)
        # define inference encoder
        self.inf_encoder_model = keras.Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_state_input_h = Input(shape=(2 * self.encoder_output_dim,))
        decoder_state_input_c = Input(shape=(2 * self.encoder_output_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)

        decoder_states = [state_h, state_c]

        decoder_outputs = TimeDistributed(decoder_dense)(decoder_outputs)
        self.inf_decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        # return all models

    def predict_sequence(self, source, n_steps):
        # encode
        state = self.inf_encoder_model.predict(source)
        intent_output = self.inf_intent_classifier.predict(source)
        # start of sequence input
        target_seq = np.zeros(shape=(1, 1, self.decoder_input_dim))
        # collect predictions
        output = list()
        for t in range(n_steps):
            # predict next char
            yhat, h, c = self.inf_decoder_model.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0, 0, :])
            # update state
            state = [h, c]
            # update target sequence
            target_seq = yhat
        return np.array(output), np.array(intent_output)

    @staticmethod
    def predict_sequence_static(top_model, source, n_steps):
        # encode
        state = top_model.inf_encoder_model.predict(source)
        intent_output = top_model.inf_intent_classifier.predict(source)
        # start of sequence input
        target_seq = np.zeros(shape=(1, 1, top_model.decoder_input_dim))
        # collect predictions
        output = list()
        for t in range(n_steps):
            # predict next char
            yhat, h, c = top_model.inf_decoder_model.predict([target_seq] + state)
            print('yhat is {}'.format(yhat.shape))
            # store prediction
            output.append(yhat[0, 0, :])
            # update state
            state = [h, c]
            # update target sequence
            target_seq = yhat
        return np.array(output), np.array(intent_output)

    def save_model_image(self):
        if self.model is None:
            print('[ERROR] Model is not defined')
            return
        plot_model(self.model, to_file='./{}.png'.format(self.model_name),
                   show_shapes=True, show_layer_names=True)

    @staticmethod
    def save_model_image_static(top_model):
        if top_model.model is None:
            print('[ERROR] Model is not defined')
            return
        plot_model(top_model.model, to_file='./{}.png'.format(top_model.model_name),
                   show_shapes=True, show_layer_names=True)

    def save_models_to_disk(self, root):
        self.model.save(root + '{}.h5'.format(self.model_name))
        self.inf_encoder_model.save(
            root + '{}_inf_encoder.h5'.format(self.model_name))
        self.inf_intent_classifier.save(
            root + '{}_inf_intent.h5'.format(self.model_name))
        self.inf_decoder_model.save(
            root + '{}_inf_decoder.h5'.format(self.model_name))

    @staticmethod
    def save_models_to_disk_static(top_model, root):
        top_model.model.save(root + '{}.h5'.format(top_model.model_name))
        top_model.inf_encoder_model.save(
            root + '{}_inf_encoder.h5'.format(top_model.model_name))
        top_model.inf_intent_classifier.save(
            root + '{}_inf_intent.h5'.format(top_model.model_name))
        top_model.inf_decoder_model.save(
            root + '{}_inf_decoder.h5'.format(top_model.model_name))

    @staticmethod
    def load_models_to_disk_static(self, root):
        self.model = lm(root + '{}.h5'.format(self.model_name))
        self.inf_encoder_model = lm(
            root + '{}_inf_encoder.h5'.format(self.model_name))
        self.inf_intent_classifier = lm(
            root + '{}_inf_intent.h5'.format(self.model_name))
        self.inf_decoder_model = lm(
            root + '{}_inf_decoder.h5'.format(self.model_name))

    def load_models_to_disk(self, root):
        self.model = lm(root + '{}.h5'.format(self.model_name))
        self.inf_encoder_model = lm(
            root + '{}_inf_encoder.h5'.format(self.model_name))
        self.inf_intent_classifier = lm(
            root + '{}_inf_intent.h5'.format(self.model_name))
        self.inf_decoder_model = lm(
            root + '{}_inf_decoder.h5'.format(self.model_name))

    @staticmethod
    def save_models_to_disk_static(top_model, root):
        top_model.model = lm(root + '{}.h5'.format(top_model.model_name))
        top_model.inf_encoder_model = lm(
            root + '{}_inf_encoder.h5'.format(top_model.model_name))
        top_model.inf_intent_classifier = lm(
            root + '{}_inf_intent.h5'.format(top_model.model_name))
        top_model.inf_decoder_model = lm(
            root + '{}_inf_decoder.h5'.format(top_model.model_name))

    def __str__(self):
        return self.model.summary()
