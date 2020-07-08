import os
import pdb
import glob
from collections import OrderedDict
import librosa
import re
from tqdm import tqdm
import datetime
from phonemizer import separator
from phonemizer.backend import FestivalBackend
from sklearn.model_selection import train_test_split


def load_wav_text(full_blizzard2013_dir, text_file, cached):
    '''
            1. load pairs of text and wav files and save to text file
            2. if cached, load saved text
    '''

    if cached:
        file_dict = text2dict(text_file)
        return file_dict
    file_dict = OrderedDict()
    backend = FestivalBackend('en-us')

    for subdir in tqdm(os.listdir(full_blizzard2013_dir)):
        full_subdir = os.path.join(full_blizzard2013_dir, subdir)
        for wav_file in glob.glob(full_subdir + '/*.wav'):
            file_name = os.path.basename(wav_file)[:-4]
            duration = get_duration(wav_file)
            txt_file = wav_file[:-3] + 'txt'
            # pdb.set_trace()
            with open(txt_file, 'r') as rfid:
                txt_content = rfid.read()
            bseparator = separator.Separator(word='#', syllable='-', phone='+')
            phoneme_sequence = backend._phonemize_aux(
                txt_content, bseparator, True)

            assert len(phoneme_sequence) == 1
            phoneme_sequence_content = phoneme_sequence[0]
            if not os.path.exists(txt_file):
                continue
            file_dict[file_name] = [
                wav_file, txt_file, phoneme_sequence_content, duration]
    dict2text(file_dict, text_file=text_file)
    return file_dict


def dict2text(dict_item, sep='|', text_file='full_blz13.txt'):
    '''
            save dictionary file to text
            key:filename
            value: list [wav_file, txt_file, duration]
    '''
    with open(text_file, 'w') as wfid:
        for key, value in dict_item.items():
            value[-1] = str(value[-1])
            value_str = '|'.join(value)
            file_line = key + '|' + value_str + '\n'
            wfid.write(file_line)


def text2dict(text_file='full_blz13.txt', sep='|'):
    '''
            recover text to dictionary 
    '''
    file_dict = OrderedDict()
    with open(text_file, 'r') as fid:
        text_lines = fid.readlines()
    for text_line in text_lines:
        file_list = re.split('\|', text_line.strip())
        # duration of speech
        file_list[-1] = float(file_list[-1])
        file_dict[file_list[0]] = file_list[1:]
    return file_dict


def get_duration(wav_file):
    """
                get duration of waveform file
    """

    audio = librosa.load(wav_file, sr=None)
    sr = audio[1]
    audio = audio[0]
    audio_len = len(audio) / sr
    return audio_len


def get_total_duration(dict_item):
    total_seconds = 0
    for key, value in dict_item.items():
        if value[-1] < 9 and value[-1] > 4:
            total_seconds += value[-1]
    return total_seconds


def get_subset(dict_item):
    """
        1. take subset of dataset which fits requirement
        2. split subset into train, valid and test
    """
    subset_list = []
    for key, value in dict_item.items():
        if value[-1] < 9 and value[-1] > 4:
            value.insert(0, key)
            value[-1] = str(value[-1])
            subset_list.append('|'.join(value) + '\n')
    train_val_test_split(subset_list, './')


def train_val_test_split(new_file_paths, out_dir):
    """
        split val test and train folder
        last step for preprocess
    """
    pdb.set_trace()
    train_set, t_test = train_test_split(
        new_file_paths, test_size=0.06, random_state=42)
    # split_t_test = len(t_test) // 2
    split_t_test = len(t_test) - 50
    valid_set = t_test[0:split_t_test]
    test_set = t_test[split_t_test:]
    set_dict = {"train": train_set, "val": valid_set, "test": test_set}
    for phase in set_dict.keys():
        write_metadata(set_dict[phase], phase, out_dir)


def write_metadata(T_list, phase, out_dir):
    output_path = os.path.join(
        out_dir, 'full_blz13_audio_text_less_9_more_4' + phase + '_filelist.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in T_list:
            f.write(line)


if __name__ == '__main__':
    full_blizzard2013_dir = "/home/gyzhang/speech_database/full-blizzard2013"
    file_dict = load_wav_text(full_blizzard2013_dir, 'full_blz13.txt', True)
    total_time = get_total_duration(file_dict)
    # print(str(datetime.timedelta(seconds=total_time)))
    get_subset(file_dict)
