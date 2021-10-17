""" This module prepares midi file data and feeds it to the neural
    network for training """
# from __future__ import absolute_import, division, print_function, unicode_literals

import numpy
from tensorflow.keras.layers import Reshape, Dense, LSTM
from tensorflow.keras.models import Sequential
from util import util
import tensorflow as tf
import pickle

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)

sequence_length = 1000
lookback = 1


def train_network():
    """ Train a Neural Network to generate music """
    notes = util.categories
    with open('data/notes', 'wb') as file_handle:
        pickle.dump(notes, file_handle)

    notes = util.get_notes()

    network_input, network_output = prepare_sequences(notes)
    model = create_network(network_input.shape, network_output.shape)
    train(model, network_input, network_output)


def prepare_sequences(notes):
    """ Prepare the sequences used by the Neural Network """

    print('preparing sequences')

    minlength = min([len(song[1]) for song in notes])
    length = int(minlength / sequence_length)

    network_input = numpy.zeros(shape=(len(notes), sequence_length - lookback, length))
    network_output = numpy.zeros(shape=(len(notes), minlength - sequence_length))

    for index, category_song in enumerate(notes):
        category, song = category_song
        song = numpy.array(list(song))
        category = numpy.array(list(category))
        while song.size % sequence_length != 0 or song.size > minlength:
            song = song[:-1, ]
        while category.size > minlength:
            category = category[:-1, ]
        song = song.reshape(sequence_length, length)

        network_input_song = song[:-lookback]
        network_input[index] = (network_input_song - network_input_song.min()) / \
                               (network_input_song.max() - network_input_song.min())
        network_output[index] = category[sequence_length * lookback:]

    network_input = network_input.astype(dtype=numpy.float32)
    network_output = network_output.astype(dtype=numpy.float32)

    return network_input, network_output


def create_network(input_shape, output_shape):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(8, input_shape=input_shape[1::], activation="relu"))
    model.add(Dense(numpy.product(output_shape[1::]), activation="softmax"))
    model.add(Reshape(output_shape[1::]))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

    return model


def store_result(model):
    model.save('model', save_format='h5')


def train(model, network_input, network_output):
    """ train the neural network """
    model.fit(network_input, network_output, epochs=1500, batch_size=5000, validation_split=.3)
    store_result(model)


train_network()
