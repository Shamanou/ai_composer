from scipy.io import wavfile
from os import listdir
from os.path import isfile, join
import numpy

categories_count = 100
categories = numpy.linspace(0, 1500, categories_count)


def get_notes():
    """ Get all the notes and chords from the wav_file files in the ./midi_songs directory """
    notes = []
    song_available = listdir("songs/")[:10]
    for index, file in enumerate(song_available):
        if isfile(join("songs/", file)):
            _, wav_file = wavfile.read("songs/" + file)

            print("Parsing song %i of %i - %s" % (index + 1, len(song_available), file))

            partitioned_by_instruments = partition_by_category(list(map(lambda x: x[0], wav_file)))
            notes.append(partitioned_by_instruments)
    return notes


def partition_by_category(wav_file):
    digitized = numpy.digitize(wav_file, categories)
    return digitized.tolist(), wav_file
