# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pickle
import numpy as np
from keras.models import load_model
import sys

from prepare_data import *

# obtain data from parameters
if len(sys.argv) < 3:
    print('Please provide arguments for this script to work. First argument should be location of file with unaccented words and morphological data '
          'and second the name of file where you would like to save file to. Example: python accentuate.py \'test_data/unaccented_dictionary\' '
          '\'test_data/accented_data\'')
    raise Exception
read_location = sys.argv[1]
write_location = sys.argv[2]

# get environment variables necessary for calculations
pickle_input = open('preprocessed_data/environment.pkl', 'rb')
environment = pickle.load(pickle_input)
dictionary = environment['dictionary']
max_word = environment['max_word']
max_num_vowels = environment['max_num_vowels']
vowels = environment['vowels']
accented_vowels = environment['accented_vowels']
feature_dictionary = environment['feature_dictionary']
syllable_dictionary = environment['syllable_dictionary']

# load models
data = Data('l', shuffle_all_inputs=False)
letter_location_model, syllable_location_model, syllabled_letters_location_model = data.load_location_models(
    'models/letters_place_20_test_epoch.h5',
    'models/syllables_place_20_test_epoch.h5',
    'models/syllabled_letters_place_20_test_epoch.h5')

letter_type_model, syllable_type_model, syllabled_letter_type_model = data.load_type_models(
    'models/letters_type_20_test_epoch.h5',
    'models/syllables_type_20_test_epoch.h5',
    'models/syllabled_letters_type_20_test_epoch.h5')

# read from data
content = data._read_content(read_location)

# format data for accentuate_word function it has to be like [['besedišči', '', 'Ncnpi', 'besedišči'], ]
content = [[el[0], '', el[1][:-1], el[0]] for el in content[:-1]]

# use environment variables and models to accentuate words
location_accented_words, accented_words = data.accentuate_word(content, letter_location_model, syllable_location_model, syllabled_letters_location_model,
                                    letter_type_model, syllable_type_model, syllabled_letter_type_model,
                                    dictionary, max_word, max_num_vowels, vowels, accented_vowels, feature_dictionary, syllable_dictionary,
                                    multext_v3=True)

# save accentuated words
with open(write_location, 'w', encoding='utf-8') as f:
    for i in range(len(location_accented_words)):
        f.write(location_accented_words[i] + '  ' + accented_words[i] + '\n')
    f.write('\n')