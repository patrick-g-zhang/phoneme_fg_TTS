import re
import os
from sklearn.model_selection import train_test_split
import sys
import pandas
import numpy as np
import pdb
import parselmouth
import pickle
from tqdm import tqdm
import librosa
from multiprocessing.pool import ThreadPool as Pool


sys.path.append("/home/gyzhang/projects/merlin/src/")
# sys.path.append("/home/gyzhang/projects/syl-embedding-tacotron2/")
from frontend.label_normalisation import HTSLabelNormalisation


class FeatureNorm(object):
    """
        This class for front end sylabel level speech synthesis with vqvae
        Functions:
        1. spawn syllable level format text input
        2. extract prosodic feature with respect to syllabel level
    """

    def __init__(self, question_file, pros_dir, wav_dir, lab_dir, out_dir):
        """
            merlin front ene question file
        """
        self.question_file = question_file  # linguistic question file
        self.pros_dir = pros_dir    # where to save or load pros feature
        self.wav_dir = wav_dir  # where to load wav_file
        self.lab_dir = lab_dir  # where to load merlin lab file
        self.out_dir = out_dir  # where to save train val test files

    def perform_vq_prosodic_front(self, ):
        """
            extract vq prosodic front feature and train-test split
            1. merge text and syl-level prosoidc feature in one string
            2. train and test split
        """
        pros_syl_list = self.perform_syl_feature_extraction()
        # pdb.set_trace()
        str_phone_list = self.perform_text_syl_word_extraction()

        # text| wave|str_phone|syl_dur_pros_path
        merged_string = ['{0}|{1}|{2}|{3}\n'.format(self.text_wav_lab_pairs[i][0], self.text_wav_lab_pairs[i]
                                                    [1], str_phone_list[i], pros_syl_list[i]) for i in range(len(self.text_wav_lab_pairs))]

        self.train_val_test_split(merged_string)

    def perform_syl_feature_extraction(self, ):
        """
            extract syl-level prosodic feature for all files
            Return:
                path list of prosodic duration information
        """
        pros_syl_list = []
        for file_index in tqdm(range(len(self.file_names))):
            file_name = self.file_names[file_index]
            wav_file = self.text_wav_lab_pairs[file_index][1]
            lab_file = self.text_wav_lab_pairs[file_index][2]
            out_file_path = os.path.join(self.pros_dir, file_name)
            # print(
            # "process syl prosodic feature extraction file name {0}".format(file_name))
            pros_dur = self.extract_syl_pros(lab_file, wav_file)
            self.save_syl_pros(pros_dur, out_file_path)
            pros_syl_list.append(out_file_path)
        return pros_syl_list

    def perform_text_syl_word_extraction(self, ):
        """
            extract syl-level text feature from all files
        """
        str_phone_list = []
        for file_index, file_name in enumerate(self.file_names):
            print(
                "process text feature extraction file name {0}".format(file_name))
            lab_file = self.text_wav_lab_pairs[file_index][2]
            str_phone = self.extract_text_syl_word(lab_file)
            str_phone_list.append(str_phone)
        return str_phone_list


class BLFeatureNorm(FeatureNorm):
    """
        relization for feature normaliztion with respect to blizzard 2013 data
    """

    def __init__(self, question_file, pros_dir, wav_dir, lab_dir, text_txt, out_dir, hop_size=0.005):
        super(BLFeatureNorm, self).__init__(
            question_file, pros_dir, wav_dir, lab_dir, out_dir)
        self.question_file = question_file
        self.hop_size = hop_size
        # get all phones in dictionary
        self.full_phone_list = self.get_phone_list(self.question_file)
        # blz13's train.txt file "/home/gyzhang/speech_database/blizzard2013/train.txt"
        self.text_txt = text_txt
        self.text_wav_lab_pairs, self.file_names = self.load_text_wav_lab_pairs()

    def select_audio(self, ):
        """
            all file list
        """
        pass

    def train_val_test_split(self, new_file_paths):
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
            self.write_metadata(set_dict[phase], phase, self.out_dir)

    def write_metadata(self, T_list, phase, out_dir):
        output_path = os.path.join(
            out_dir, 'blz13_audio_text_' + phase + '_filelist.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in T_list:
                f.write(line)

    def valid_audio(self, wav_file):
        """
            valid duration of audio
            drop audio too long or too short
        """

        audio = librosa.load(wav_file, sr=None)
        sr = audio[1]
        audio = audio[0]
        audio_len = len(audio) / sr
        if audio_len > 20 or audio_len < 1:
            return False
        else:
            return True

    def load_text_wav_lab_pairs(self, ):
        """
            1. load all text and wav and lab file pairs in corpus
            2. ignore some files which doesn't exist
            3. file_names list
            Returns:
                text_wav_labs: text wav and lab file paths
                file_names: file names list
        """
        text_wav_labs = []
        file_names = []
        with open(self.text_txt, 'r') as fid:
            file_paths = fid.readlines()
        for file_path in file_paths:
            _, _, _, text, audio_path = re.split("\|", file_path.strip())

            # get filename
            wav_file_name = os.path.basename(audio_path)

            # valid audio's length, we drop audio whose length >20s or <1s
            if not self.valid_audio(audio_path):
                continue

            # .wav -> ".lab"
            lab_file_name = re.sub("wav", "lab", wav_file_name)
            lab_file_path = os.path.join(self.lab_dir, lab_file_name)

            # some lab files might not exists due to alignment
            if not os.path.exists(lab_file_path):
                print("lab file doesn't exist {0}".format(lab_file_path))
                continue
            assert os.path.exists(audio_path)
            file_names.append(re.split("\.", wav_file_name)[0])
            text_wav_labs.append((text, audio_path, lab_file_path))

        return text_wav_labs, file_names

    def get_phone_list(self, question_file):
        """
            get all phones in dictionary with question file
            Args:
                question file: merlin question file
            Returns:
                full_phone_list : all phone's symbols
        """
        df_qs = pandas.read_csv(question_file, sep="\s+",
                                header=None).drop(columns=[0, 2])

        # all phones in order
        full_phone_list = [re.split(
            "\-", cphone)[1] for cphone in df_qs.values.reshape(1, -1)[0].tolist()[0:49]]
        return full_phone_list

    def extract_text_syl_word(self, lab_file):
        """
            1. extract numerical infos from lab file
            2. generate word-syl-phone list
            3. convert it to string
            Args:
                lab_file which contains festival process information
            Returns:
                string contains information of given list
        """
        phone_matrix, syl_matrix, syl_stress, word_syl, c_vowel, _ = self.search_from_lab_file(
            lab_file)

        phone_syl = self.get_phone_syl_word(
            phone_matrix, syl_matrix, syl_stress, word_syl, c_vowel)

        str_phone = self.phone_syl_to_string(phone_syl)
        return str_phone

    def phone_syl_to_string(self, phone_syl):
        """
            we need convert list to string for store
            phone delimiter: +
            syl_delimiter: -
            word_delimiter: #
            Args:
                phone_syl: hierachy structure of phone-syl-word in list
            Returns:
                string contains information of given list
        """

        new_line = []
        for syl_list in phone_syl:
            new_syl_list = []
            for phone_list in syl_list:
                phone_line = "+".join(phone_list)
                new_syl_list.append(phone_line)
            syl_line = "-".join(new_syl_list)
            new_line.append(syl_line)
        str_phone = "#".join(new_line)
        return str_phone

    def get_phone_syl_word(self, phone_matrix, syl_matrix, syl_stress, word_syl, c_vowel):
        """
            extract phone-syl-word structure from lab file
            example:
            autobiography -> [[['AO'], ['T', 'AX'], ['B', 'AY'], ['AA'], ['G', 'R', 'AX'], ['F', 'IY']]]
            Args:
                phone_matrix: binary phone matrix
                syl_matrix: number of phone in one syl
                word_syl: number of syl in one word
            Returns:phone_syl like example
        """
        # find phone_index from phone binary matrix
        try:
            phone_index_seqs = np.squeeze(np.apply_along_axis(
                lambda x: np.argwhere(x == 1.0), 1, phone_matrix)).tolist()
        except:
            print("Value Error of lab file")
            pdb.set_trace()

        phone_seqs = [self.full_phone_list[i].upper()
                      for i in phone_index_seqs]  # ['SILENCES', 'S', 'AH', 'M',]

        phone_index = 0
        phone_syl = []
        syl_matrix_list = syl_matrix.tolist()
        while phone_index < len(phone_seqs):
            # start with a new word
            # syllabel number of a word
            syls_in_word_nums = word_syl[phone_index]
            if syls_in_word_nums == -1.0:
                # show it should be silence or pauses, jump
                phone_index += 1
                continue

            syls_in_word = []  # all syls in this word
            for i in range(int(syls_in_word_nums)):
                # phone in this syl
                syl_num = int(syl_matrix_list[phone_index])
                if syl_num == -1:
                    phone_index += 1
                else:
                    # add one syl's phones sequence
                    phones_in_syl = []
                    for phone_syl_index in range(phone_index, phone_index + syl_num):
                        cur_phone = phone_seqs[phone_syl_index]
                        if c_vowel[phone_syl_index] == 1.0:
                            cur_phone = cur_phone + \
                                '1' if syl_stress[phone_syl_index] == 1.0 else cur_phone + '0'
                        phones_in_syl.append(cur_phone)
                    phone_index += syl_num
                    syls_in_word.append(phones_in_syl)
#             phone_index += syl_num
            phone_syl.append(syls_in_word)
        return phone_syl

    def extract_syl_pros(self, lab_file, wav_file):
        """
            1. extract dur_feature_matrix and syl_matrix from lab_file
            2. get dur infos for each syllabel
            3. extract f0 and intensity infos from wav_file
            4. align duration and prosodic feature
            Args:
                lab_file: merlin lab file
                wav_file: wav
            Returns:
                pros_syls: prosody information for each syl
                dur_syl: dur information for each syl
        """
        _, syl_matrix, _, _, _, dur_feature_matrix = self.search_from_lab_file(
            lab_file)
        syl_dur = self.get_syl_dur(dur_feature_matrix, syl_matrix)
        pitch_int, intensity = self.extract_f0_intensity(wav_file)
        pros_syls, dur_syls = self.align_prosodic_dur(
            syl_dur, pitch_int, intensity)
        return [pros_syls, dur_syls]

    def save_syl_pros(self, pros_dur, out_file_path):
        """
            save pros and duration information to out file 
            Args:
                pros_dur: pros and dur infos [pros_syls, dur_syls]
        """
        with open(out_file_path, "wb") as fid:
            pickle.dump(pros_dur, fid)

    def search_from_lab_file(self, lab_file):
        """
            search information from merlin linguistic label file
            Args:
                lab file: merlin linguistic label file
                qustion file: merlin english question file
            Returns:
                infos in merlin label file in numerical format
        """
        label_normaliser = HTSLabelNormalisation(
            question_file_name=question_file, add_frame_features=False, subphone_feats='none')

        label_feature_matrix = label_normaliser.load_labels_with_phone_alignment(
            lab_file, None)
        if not len(label_feature_matrix.tolist()):
            print("Found empty lab file {0}".format(lab_file))
            exit(1)

        # phone infos of this lab file
        phone_matrix = label_feature_matrix[:, 0:48]        # phone infos
        syl_matrix = label_feature_matrix[:, 50]            # sylable infos
        syl_stress = label_feature_matrix[:,
                                          51].tolist()   # sylable stress infos
        word_syl = label_feature_matrix[:, 52].tolist()     # word infos
        c_vowel = label_feature_matrix[:, 49].tolist()      # current vowel

        # dur infos
        dur_feature_matrix = label_normaliser.extract_dur_from_phone_alignment_labels(
            lab_file, "numerical", "phoneme", "phoneme")

        return phone_matrix, syl_matrix, syl_stress, word_syl, c_vowel, dur_feature_matrix

    def get_syl_dur(self, dur_feature_matrix, syl_matrix):
        """
            get non-silence's syllabel's dur (start end time) which will serve for prosodic feature extraction 
            for each syl
            Args:
                dur_feature_matrix: dur infos in one sentence
                syl_matrix: syl infos 
            Returns: dur{start and end time } of each syllabel
            [[0.255, 0.455],
            [0.455, 0.6900000000000001],
            [0.6900000000000001, 0.81],
            [0.81, 1.18]]
        """
        dur_feature_list = np.squeeze(dur_feature_matrix).tolist()
        syl_matrix_list = syl_matrix.tolist()

        phone_index = 0
        syl_dur = []
        start_frame_index = 0
        while phone_index < len(syl_matrix_list):
            # syl_num related to current phone index
            syl_num = int(syl_matrix_list[phone_index])
            if syl_num == -1:
                # start_time add silence
                frame_number = dur_feature_list[phone_index]
                start_frame_index += frame_number
                # silence phone
                phone_index += 1
            else:
                end_frame_index = start_frame_index
                for phone_syl_index in range(phone_index, phone_index + syl_num):
                    end_frame_index = end_frame_index + \
                        dur_feature_list[phone_syl_index]
                syl_dur.append([start_frame_index * self.hop_size,
                                end_frame_index * self.hop_size])
                start_frame_index = end_frame_index
                phone_index += syl_num
        return syl_dur

    def extract_f0_intensity(self, wav_file):
        """
            1. extract f0 and intensity from wav file
            2. align between f0 and intensity 
            because algorithm extracting f0 and intensity using different window
            f0 is 0.004 and intensity is 0.008. intensity has fewer frame number.
            we align f0 to intensity resolution
            Args:
                wav_file: speech file 
            Returns:
                pitch_int: pitch and intensity feature [2, int_frame_number]
                intensity: intensity object
        """
        snd = parselmouth.Sound(wav_file)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75)
        pitch_values = pitch.selected_array['frequency']
        intensity = snd.to_intensity(minimum_pitch=75, time_step=0.01)
        intensity_frame_number = intensity.get_number_of_frames()
        # pitch and intensity arrary with the same length with intensity
        # window for pitch 0.04 window for intensity 0.08
        pitch_int = np.zeros((2, intensity_frame_number))
        pitch_start = 2
        pitch_end = pitch_start + intensity_frame_number
        pitch_int[0, :] = pitch_values[pitch_start:pitch_end]
        pitch_int[1, :] = np.squeeze(intensity.values)
        return pitch_int, intensity

    def align_prosodic_dur(self, syl_dur, pitch_int, intensity):
        """
            align duration and prosodic feature
            because duration infos spawn from htk tools while prosodic extracted by praat
            we need to align it
            Args:
                syl_dur: dur{start and end time } of each syllabel
                pitch_int: pitch and intensity feature [2, int_frame_number]
                intensity: intensity object
            Returns:
                pros_syls: list of prosodic feature [syl_len, 2] for each syllabel
                dur_syls: list of duration of each syllabel
        """
        pros_syls = []
        dur_syls = []
        for per_syl_dur in syl_dur:
            start_time_frame = round(
                intensity.get_frame_number_from_time(per_syl_dur[0]))
            end_time_frame = round(
                intensity.get_frame_number_from_time(per_syl_dur[1]))
            pros_syl = pitch_int[:, start_time_frame - 1:end_time_frame - 1]
            pros_syls.append(pros_syl.transpose())
            dur_syls.append(end_time_frame - start_time_frame)
        return pros_syls, dur_syls

    def save_prosodic_dur(self, pros_syls, dur_syls, out_file):
        """
            dump pros and dur list to json file
        """
        with open(out_file, 'wb') as fp:
            pickle.dump([pros_syls, dur_syls], fp)


if __name__ == "__main__":
    question_file = './syl_embedding_tac.hed'
    lab_file_dir = "/home/gyzhang/merlin/egs/blizzard2013/s1/database/labels/label_phone_align"
    pros_dir = "/home/gyzhang/projects/syl-embedding-tacotron2/datasets/blizzard2013/pros_dur"
    wav_dir = "/home/gyzhang/speech_database/blizzard2013/segmented/wavn"
    out_dir = "./"
    text_txt = "/home/gyzhang/speech_database/blizzard2013/train.txt"
    if not os.path.exists(pros_dir):
        os.mkdir(pros_dir)
    blf = BLFeatureNorm(question_file, pros_dir,
                        wav_dir, lab_file_dir, text_txt, out_dir)
    blf.perform_vq_prosodic_front()