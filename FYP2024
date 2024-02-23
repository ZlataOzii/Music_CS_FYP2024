import os
import json
import music21 as m21
from music21 import *
import tensorflow.keras as keras
import numpy as np

# change path in main function and .mid in load_music function
MIDI_DATASET_PATH = "classical/unzip/test_files"
# change path in main function and .krn in load_music function
KRN_DATASET_PATH = "archive/test"
SAVE_DIR = "archive/musicnet/musicnet/test_data/MIDI/dataset"
UNITE_DATA = "archive/unitedata"
SEQ_LEN = 64
MAPPING_PATH = "mapping.json"

# pre-processing the data

# acceptable notes in the notesheet
"""
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]
"""

def load_music(dataset_path):

    songs = []

    # go through all the files in the dataset,  load and convert them to be written in a list called "songs"
    for path, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".krn"):
                song = converter.parse(os.path.join(path, file))
                songs.append(song)
                print(file)
    print(songs)
    return songs



def has_acceptable_durations(song, accept_dur):
    # for note in song.flat.notesAndRests:
    for note in song.flatten().notesAndRests:
        if note.duration.quarterLength not in accept_dur:
            return False
    return True


def transpose(song):
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]


    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key") # a tool to predict the key from the piece, built in music21 library

    print(key)
    # get interval for transpositions E.g. Bmaj ->Cmaj. Both Cmaj and Amin do not have any additional notations
    if key.mode =="major":
        #if the key is major, convert it to C major
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        #if the key is minor, convert it to A minor
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song

def encode_song(song, time_step=0.25):

    encoded_song = []
    # pitch = 60, duration = 1.0 -> [60, "_", "_", "_"]

    for event in song.flatten().notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        #handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        #convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            # if it's the first time we see a note/rest, let's encode it. Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast the encoded song into the string
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

def preprocess(dataset_path):
    pass

    # load the midi files
    print("Loading songs...")
    songs = load_music(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):
    # filter non-acceptable dur
        #if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            #continue
    # transpose to C major or A minor
        song = transpose(song)
    # encode songs with music time series representation
        encoded_song = encode_song(song)
    # save midi files to simple text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def unite_data(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]

    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs

    # save string that contains all dataset

def create_mapping(songs, mapping_path):
    mappings = {}

    # identify vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save vocabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent = 4)

def convert_songs_to_int(songs):
    int_songs =[]

    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # cast songs string to a list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generate_training_seq(seq_len):

    # load songs and map them to int
    songs = load(UNITE_DATA)
    int_songs = convert_songs_to_int(songs)

    # generate the training seq
    inputs= []
    targets= []

    seq_num = len(int_songs) - seq_len
    for i in range(seq_num):
        inputs.append(int_songs[i:i+seq_len])
        targets.append(int_songs[i+seq_len])

    # one-hot encode the sequence
    # inputs: (# of seq, seq len=64 vocabulary size)
    # [ [0, 1, 2] [1, 1, 2] ] ->  [ [ [1, 0, 0], [0, 1, 0], [0, 0, 1],] [] ]
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to.categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets

def main():
    preprocess(KRN_DATASET_PATH)
    songs = unite_data(SAVE_DIR, UNITE_DATA, SEQ_LEN)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_seq(SEQ_LEN)
    a = 1

if __name__ == "__main__":
    main()
    #songs = load_music(KRN_DATASET_PATH)
    #print(f"Loaded {len(songs)} songs.")
    #song = songs[0]



    # transpose the song
    #transposed_song = transpose(song)
    #transposed_song.show()








