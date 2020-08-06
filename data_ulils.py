import torch
import random
import numpy as np
import re
from hparams import letters
import librosa
from utils import dynamic_range_compression, dynamic_range_decompression
from utils import load_wav_to_torch
import layers

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(letters)}
_id_to_symbol = {i: s for i, s in enumerate(letters)}
_whitespace_re = re.compile(r'\s+')
_symbols = re.compile(r'[\!\'\(\)\,\.\:\;\?\-]')


def text_to_sequence(audiopath, text, word_to_audio, audio_to_sentences, glued_num):
    sequence = []
    glued_sequence = []
    audio_list = []
    for s in text:
        if s in _symbol_to_id:
            sequence += [_symbol_to_id[s]]

    for word in text.split():
        # avoid less than N audios
        if len(word_to_audio[word]) > glued_num:
            audio_set = random.sample(word_to_audio[word], glued_num)
        else:
            audio_set = word_to_audio[word]
        # print(word, word_to_audio[word])
        audio_list += audio_set
    # avoid repeated audio
    audio_list = list(set(audio_list))
    # avoid target audio
    if len(audio_list) >1:
        audio_list = list(filter(lambda a: a != audiopath, audio_list))
    # print(audio_list)
    for path in audio_list:
        for s in audio_to_sentences[path]:
            if s in _symbol_to_id:
                glued_sequence += [_symbol_to_id[s]]
    return sequence, glued_sequence, audio_list


def load_filepaths_and_text(filename, split='|'):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
        return filepaths_and_text


def produce_inverted_index(audiopaths_and_texts):
    print("Generate inverted index...")
    audio_to_words = {}
    audio_to_sentences = {}
    words_set = []
    for audiopaths_and_text in audiopaths_and_texts:
        audio_path, text = audiopaths_and_text[0], audiopaths_and_text[1]
        text = text.lower()
        text = re.sub(_symbols, ' ', text)
        audio_to_words[audio_path] = text.split()
        audio_to_sentences[audio_path] = text
        words_set += text.split()
    words_set = set(words_set)
    word_to_audios = {}
    for audio_path, word_list in audio_to_words.items():
        for word in set(word_list):
            if word not in word_to_audios:
                word_to_audios[word] = [audio_path]
            elif word in word_to_audios:
                word_to_audios[word] += [audio_path]
    print("Generate done!")
    return word_to_audios, audio_to_sentences


class TextMelLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.win_length = hparams.win_length
        self.hop_length = hparams.hop_length
        self.n_mel_channels = hparams.n_mel_channels
        self.mel_fmin = hparams.mel_fmin
        self.mel_fmax = hparams.mel_fmax
        self.word_to_audio, self.audio_to_sentences = produce_inverted_index(self.audiopaths_and_text)
        self.glued_num = hparams.glued_num
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        # preprocess sentence
        text = text.lower()
        text = re.sub(_symbols, ' ', text)
        text = re.sub(_whitespace_re, ' ', text)

        text, glued_text, audio_list = self.get_text(audiopath, text)
        mel = self.get_mel(audiopath)
        glued_mel = []
        # print(audio_list)
        for audio in audio_list:
            glued_mel += [self.get_mel(audio)]
        glued_mel = torch.cat(glued_mel, -1)
        return (text, glued_text, mel, glued_mel)

    def get_mel(self, filename):
        # produce target mel spectral features
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        # melspec = librosa.feature.melspectrogram(y=audio, sr=sampling_rate,
        #                                          n_fft=self.filter_length, hop_length=self.hop_length, power=1,
        #                                          n_mels=self.n_mel_channels, fmin=self.mel_fmin, fmax=self.mel_fmax)
        # melspec_features = dynamic_range_compression(torch.FloatTensor(melspec.astype(np.float32)))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec

    def get_text(self, audiopath, text):
        text_norm, glued_text_norm, audio_list = \
            text_to_sequence(audiopath, text, self.word_to_audio, self.audio_to_sentences, self.glued_num)
        return torch.IntTensor(text_norm), torch.IntTensor(glued_text_norm), audio_list

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self):
        super(TextMelCollate, self).__init__()

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        # pad glued on-hot text sequence
        max_glued_text_len = batch[ids_sorted_decreasing[0]][1].size(0)
        glued_text_padded = torch.LongTensor(len(batch), max_glued_text_len)
        glued_text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)-1, -1, -1):
            text = batch[ids_sorted_decreasing[i]][1]
            if i is not len(ids_sorted_decreasing)-1:
                if text.size(0) > max_glued_text_len:
                    glued_text_padded[i, :] = text[:max_glued_text_len]
                else:
                    glued_text_padded[i, :text.size(0)] = text
                pre_text = batch[ids_sorted_decreasing[i+1]][1]
                if text.size(0) < pre_text.size(0) < max_glued_text_len:
                    glued_text_padded[i+1, :].zero_()
                    glued_text_padded[i+1, :text.size(0)] = pre_text[:text.size(0)]
            else:
                if text.size(0) > max_glued_text_len:
                    glued_text_padded[i, :] = text[:max_glued_text_len]
                else:
                    glued_text_padded[i, :text.size(0)] = text

        # pad glued mel sequence
        max_glued_mel_len = batch[ids_sorted_decreasing[0]][3].size(1)
        glued_mel_padded = torch.FloatTensor(len(batch), num_mels, max_glued_mel_len)
        glued_mel_padded.zero_()
        for i in range(len(ids_sorted_decreasing)-1, -1, -1):
            mel = batch[ids_sorted_decreasing[i]][3]
            if i is not len(ids_sorted_decreasing)-1:
                if mel.size(1) > max_glued_mel_len:
                    glued_mel_padded[i, :, :] = mel[:, :max_glued_mel_len]
                else:
                    glued_mel_padded[i, :, :mel.size(1)] = mel
                pre_mel = batch[ids_sorted_decreasing[i + 1]][3]
                if mel.size(1) < pre_mel.size(1) < max_glued_mel_len:
                    glued_mel_padded[i + 1, :, :].zero_()
                    glued_mel_padded[i + 1, :, :mel.size(1)] = pre_mel[:, :mel.size(1)]
            else:
                if mel.size(1) > max_glued_mel_len:
                    glued_mel_padded[i, :, :] = mel[:, :max_glued_mel_len]
                else:
                    glued_mel_padded[i, :, :mel.size(1)] = mel

            # mel = batch[ids_sorted_decreasing[i]][2]
            # glued_mel_padded[i, :, :mel.size(1)] = mel

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, \
               glued_text_padded, glued_mel_padded


def get_mel(filename, hparams):
        audio, sampling_rate = librosa.core.load(filename)
        if sampling_rate != hparams.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, hparams.sampling_rate))
        melspec = librosa.feature.melspectrogram(y=audio, sr=sampling_rate,
                                                 n_fft=hparams.filter_length, hop_length=hparams.hop_length, power=1,
                                                 n_mels=hparams.n_mel_channels, fmin=hparams.mel_fmin, fmax=hparams.mel_fmax)
        melspec_features = dynamic_range_compression(torch.FloatTensor(melspec.astype(np.float32)))
        return melspec_features


def get_mel_text_pair_inference(text, hparams):
    audiopaths_and_text = load_filepaths_and_text(hparams.training_files)
    word_to_audio, audio_to_sentences = produce_inverted_index(audiopaths_and_text)
    # preprocess sentence
    text = text.lower()
    text = re.sub(_symbols, ' ', text)
    text = re.sub(_whitespace_re, ' ', text)

    text_norm, glued_text_norm, audio_list = \
        text_to_sequence(audiopaths_and_text, text, word_to_audio, audio_to_sentences, hparams.glued_num)
    text_norm = torch.IntTensor(text_norm)
    glued_text_norm = torch.IntTensor(glued_text_norm)

    glued_mel = []
    for audio in audio_list:
        glued_mel += [get_mel(audio, hparams)]
    glued_mel = torch.cat(glued_mel, -1)
    return (text_norm.unsqueeze(0), glued_text_norm.unsqueeze(0), glued_mel.unsqueeze(0))


if __name__ == "__main__":
    y, sample_rate = librosa.core.load("LJ001-0002.wav", sr=22050)
    # magnitudes1 = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=1024, hop_length=256, power=1)
    # melspectrogram = librosa.feature.melspectrogram(S=magnitudes1, n_fft=1024, hop_length=256, power=2.0)

    print('melspectrogram.shape', melspectrogram.shape)
    print(melspectrogram)

    audio_signal = librosa.feature.inverse.mel_to_audio(melspectrogram, sr=22050, n_fft=1024, hop_length=256, power=2)
    print(audio_signal, audio_signal.shape)

    librosa.output.write_wav('test2.wav', audio_signal, 22050)
