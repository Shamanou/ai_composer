from music21 import converter, instrument, note, chord
from os import listdir
from os.path import isfile, join


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    for file in listdir("src/midi_songs/"):
        if isfile(join("src/midi_songs/", file)):
            midi = converter.parse("src/midi_songs/" + file)

            print("Parsing %s" % file)

            notes_to_parse = []

            partitioned_by_instruments = instrument.partitionByInstrument(midi)
            for instrument_element_in_mid in partitioned_by_instruments.parts:
                tmp = []
                for instrument_in_mid in instrument_element_in_mid:
                    tmp.append(instrument_in_mid)
                notes_to_parse.append(tmp)

            for parsed_instrument in notes_to_parse:
                tmp = []
                for element in parsed_instrument:
                    if isinstance(element, note.Note):
                        tmp.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        tmp.append('.'.join(str(n) for n in element.normalOrder))
                notes.append(tmp)
    return notes
