# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys
sys.path.insert(0, '../../../')
from prepare_data import *

import pickle
# from keras import backend as Input
np.random.seed(7)


# obtain data from parameters
if len(sys.argv) < 3:
    print('Please provide arguments for this script to work. First argument should be location of file with unaccented words and morphological data, '
          'second the name of file where you would like to save results to and third location of ReLDI tagger. Example: python accentuate.py '
          '\'test_data/original_connected_text\' \'test_data/accented_connected_text\' \'../reldi_tagger\'')
    raise Exception
read_location = sys.argv[1]
write_location = sys.argv[2]
reldi_location = sys.argv[3]


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

# get models
data = Data('l', shuffle_all_inputs=False)
letter_location_model, syllable_location_model, syllabled_letters_location_model = data.load_location_models(
    'models/letters_place_20_test_epoch.h5',
    'models/syllables_place_20_test_epoch.h5',
    'models/syllabled_letters_place_20_test_epoch.h5')

letter_type_model, syllable_type_model, syllabled_letter_type_model = data.load_type_models(
    'models/letters_type_20_test_epoch.h5',
    'models/syllables_type_20_test_epoch.h5',
    'models/syllabled_letters_type_20_test_epoch.h5')

# get word tags
tagged_words, original_text = data.tag_words(reldi_location, read_location)


# find accentuation locations
predictions = data.get_ensemble_location_predictions(tagged_words, letter_location_model, syllable_location_model,
                                                     syllabled_letters_location_model,
                                                     dictionary, max_word, max_num_vowels, vowels, accented_vowels, feature_dictionary,
                                                     syllable_dictionary)

location_accented_text = data.create_connected_text_locations(tagged_words, original_text, predictions, vowels)

# accentuate text
location_y = np.around(predictions)
type_predictions = data.get_ensemble_type_predictions(tagged_words, location_y, letter_type_model, syllable_type_model,
                                                      syllabled_letter_type_model,
                                                      dictionary, max_word, max_num_vowels, vowels, accented_vowels, feature_dictionary,
                                                      syllable_dictionary)

accented_text = data.create_connected_text_accented(tagged_words, original_text, type_predictions, location_y, vowels, accented_vowels)

# save accentuated text
with open(write_location, 'w', encoding='utf-8') as f:
    f.write(accented_text)
