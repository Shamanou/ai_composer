from scipy.io import wavfile
from os import listdir
from os.path import isfile, join

categories_count = 100
categories = list(range(categories_count))


def get_notes():
    """ Get all the notes and chords from the wav_file files in the ./midi_songs directory """
    notes = []
    for file in listdir("src/songs/"):
        if isfile(join("src/songs/", file)):
            _, wav_file = wavfile.read("src/songs/" + file)

            print("Parsing %s" % file)

            partitioned_by_instruments = partition_by_category(wav_file[0])
            notes.append(partitioned_by_instruments)
    return notes


def partition_by_category(wav_file):
    categorized_wav_file = []
    for entry in wav_file:
        for category in categories:
            if entry <= category:
                categorized_wav_file.append(category)
                break
    return categorized_wav_file
