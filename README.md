# Introduction
There is no simple algorithm for stress assignment of Slovene words. Speakers of Slovene are usually taught accents together with words. Machine learning algorithms give positive results on this problem, therefore we tried deep neural networks. We tested different architectures, data presentations and an ensemble of networks. We achieved the best results using the ensemble method, which correctly predicted 87.62 % of tested words. Our neural network approach improved results of other machine learning methods and proved to be successful in stress assignment.

This is a shorter version of master thesis project (whole source code is published on https://github.com/lkrsnik/accetuation). This shorter version does not contain results from tests and is slightly simplified in comparison with the whole project. It contains scripts for accentuation of words from simple list, accentuation application for connected text and a simple example script of learning neural network (we do not provide learning data for this, due to copy rights, if you want it to work, you have to obtain your own learning set).

# Set up
The majority of programs used in this app are easily installable with following command:
```
pip install -r requiremets.txt
```
If you encounter any problems while installing Keras you should check out their official site (https://keras.io/#installation). The results from neural networks were trained on Theano backend. Although TensorFlow might work as well it is not guaranted.

# Structure
This repository contains three folders models, where best preforming weights are saved, preprocessed_data - this contains pickled data with enviromental constants and test_data, with examples data. Repository also contains four important files, prepare_data.py with majority of code, accentuate.py, meant as simple accentuation app, accentuate_connected_text.py - simple app for accentuating connected Slovene text - and learn_location_weights, meant as an example of how you could learn your own neural networks with different parameters.

# accentuate.py
You should use this script, if you would like to accentuate words from list of words with their morphological data. For it to work you should generate file, which in each line contains word of interest and morphological data, separated by tab. It should look like this:
```
absolutistični	Afpmsay-n
spoštljivejše	Afcfsg
tresoče	Afpfsg
raznesena	Vmp--sfp
žvižgih	Ncmdl

```
You can call this script in bash with following command:
```
python accentuate.py <path_to_input_file> <path_to_results_file>
```
Here is a working example:
```
python accentuate.py 'test_data/unaccented_dictionary' 'test_data/accented_data'
```

# accentuate_connected_text.py
This app uses external tagger for obtaining morphological information from sentences. For it to work you should clone repository from https://github.com/clarinsi/reldi-tagger and pass its location as a parameter.
```
python accentuate_connected_text.py <path_to_input_file> <path_to_results_file> <path_to_reldi_repository>
```
You can try working example with your actual path to reldi:
```
python accentuate_connected_text.py 'test_data/original_connected_text' 'test_data/accented_connected_text' '../reldi_tagger'
```

# learn_location_weights.py
This is an example of script designed for learning weights. For it to work you have to have learning data. Given example can be used for learning neural networks for assigning location of stress from letters. For examples of other neural networks you should look into original repository.
