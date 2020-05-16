""" This module prepares midi file data and feeds it to the neural
    network for training """
# from __future__ import absolute_import, division, print_function, unicode_literals

import numpy
from tensorflow.keras.layers import Reshape, Dense, Dropout, LSTM, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from util import Util
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import pickle

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)


def train_network():
    """ Train a Neural Network to generate music """
    notes = Util.get_notes()
    with open('src/data/notes', 'wb') as file_handle:
        pickle.dump(notes, file_handle)

    # get amount of pitch names
    n_vocab = [len(set(instrument)) for instrument in notes]

    with open('src/data/instruments','wb') as file_handle:
        pickle.dump(n_vocab, file_handle)

    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input.shape, network_output.shape)
    train(model, network_input, network_output)


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitch_names = []
    for instrument_in_mid in notes:
        pitch_names.append(sorted(set(item for item in instrument_in_mid)))

    # create a dictionary to map pitches to integers
    note_to_int = []
    for instrument_in_mid in pitch_names:
        note_to_int.append(dict((note, number) for number, note in enumerate(instrument_in_mid)))

    network_input = [[]] * len(notes)
    network_output = [[]] * len(notes)

    # create input sequences and the corresponding outputs
    for instrument_index, instrument_in_mid in enumerate(notes):
        for i in range(0, len(instrument_in_mid) - sequence_length, 1):
            sequence_in = instrument_in_mid[i:i + sequence_length]
            sequence_out = instrument_in_mid[i + sequence_length]
            network_input[instrument_index].append([note_to_int[instrument_index][char]
                                                    for char in sequence_in])
            network_output[instrument_index].append(note_to_int[instrument_index][sequence_out])

    network_input = numpy.array(network_input, dtype=numpy.float32)
    network_input = network_input.reshape((network_input.shape[1], -1, network_input.shape[0]))
    network_output = numpy.array(network_output, dtype=numpy.float32)
    network_output = network_output.reshape((network_output.shape[1], network_output.shape[0], -1))

    # normalize input
    for i, vocab in enumerate(n_vocab):
        network_input[i] = network_input[i] / (vocab + 1)

    network_output = to_categorical(network_output)
    network_output = network_output.reshape((network_output.shape[0], network_output.shape[2], network_output.shape[1]))
    return network_input, network_output


def create_network(input_shape, output_shape):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape[1::], return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(1024))
    model.add(Dense(numpy.product(output_shape[1::]), activation="relu"))
    model.add(Reshape(output_shape[1::]))
    model.add(Dropout(0.1))
    model.add(Activation('softmax'))
    optimizer = SGD(lr=0.01, momentum=0.9, clipnorm=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def store_result(model):
    model.save('model', save_format='h5')


def train(model, network_input, network_output):
    """ train the neural network """
    model.fit(network_input, network_output, epochs=300, batch_size=300)
    store_result(model)


train_network()
