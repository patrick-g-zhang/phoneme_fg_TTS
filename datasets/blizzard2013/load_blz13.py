from text import *
from text.symbols import en_symbols, _punctuation

_symbol_to_id = {s: i for i, s in enumerate(en_symbols)}


class BLLoadFeature():
    """
        this class is for load feature for dataset blizzard 2013
    """

    def __init__(self, ):
        pass

    def phone_to_sequence(self, phone_list):
        """
            translate syl phone to sequence
        """
        sequence = []
        for phone in phone_list:
            tid = _symbol_to_id[phone]
            sequence.append(tid)
        return sequence

    def load_phoneme_text(self, str_phone):
        phone_list = []
        phone_list.append('SOS')
        for in_syl in re.split("\#", str_phone):
            syl_phone_list = []
            for in_phone in re.split("\-", in_syl):
                syl_phone_list.extend(re.split("\+", in_phone))
            phone_list.extend(syl_phone_list)
            phone_list.append(' ')
        phone_list = phone_list[:-1]
        phone_list.append('EOS')
        return phone_list
