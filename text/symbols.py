""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = [s for s in cmudict.festival_valid_symbols]
_start_end = ['SOS', 'EOS']

# Export all english symbols:
en_symbols = [_pad] + _start_end + list(_special) + \
    list(_punctuation) + _arpabet
