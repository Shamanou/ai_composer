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
    notes = Util.categories
    with open('data/notes', 'wb') as file_handle:
        pickle.dump(notes, file_handle)

    # get amount of pitch names
    n_vocab = Util.categories_count

    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input.shape, network_output.shape)
    train(model, network_input, network_output)


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # create a dictionary to map pitches to integers
    note_to_int = Util.get_notes()

    network_input = get_empty_notes_list(notes)
    network_output = get_empty_notes_list(notes)

    # create input sequences and the corresponding outputs
    for index, song in enumerate(notes):
        for i in range(0, len(song) - sequence_length, 1):
            sequence_in = song[i:i + sequence_length]
            sequence_in = map(lambda x: x[1], sequence_in)
            sequence_out = song[i + sequence_length]
            sequence_out = map(lambda x: x[0], sequence_out)
            network_input[index].append(sequence_in)
            network_output[index].append(sequence_out)

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


def get_empty_notes_list(notes):
    return [[]] * len(notes)


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
