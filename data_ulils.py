import torch
import random
from scipy.io.wavfile import read
import numpy as np
from stft import STFT
from unidecode import unidecode
import re
from hparams import symbols
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression
import python_speech_features

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text):
    sequence = []
    text = unidecode(text)
    text = text.lower()
    _whitespace_re = re.compile(r'\s+')
    text = re.sub(_whitespace_re, ' ', text)
    for s in text:
        if s in _symbol_to_id and s is not '_' and s is not '~':
            sequence += [_symbol_to_id[s]]
    return sequence


def load_filepaths_and_text(filename, split='|'):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
        return filepaths_and_text


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


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
        self.stft = ConcateSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        audio1, sampling_rate1 = torchaudio.load(filename)
        if sampling_rate1 != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate1, self.sampling_rate))
        # window = torch.hann_window(self.filter_length)
        # specgram = torchaudio.transforms.Spectrogram(win_length=self.win_length, power=2,
        #                                              hop_length=self.hop_length, n_fft=self.filter_length)(audio1)
        # melspec1 = torchaudio.transforms.MelScale(n_mels=self.n_mel_channels, sample_rate=self.sampling_rate,
        #                                           f_min=self.mel_fmin, f_max=self.mel_fmax,
        #                                           n_stft=self.filter_length // 2 + 1)(specgram)
        # melspec1 = torchaudio.transforms.MelSpectrogram(sample_rate=self.sampling_rate, win_length=self.win_length, hop_length=self.hop_length, )(audio1)
        stft = torch.stft(audio1, hop_length=self.hop_length, n_fft=self.filter_length, win_length=self.win_length, window=torch.hann_window(self.win_length))
        spectro_torch = stft.pow(2).sum(-1)
        melspec1 = torchaudio.transforms.MelScale(n_mels=self.n_mel_channels, sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                  f_max=self.mel_fmax)(spectro_torch)
        melspec1 = torch.squeeze(melspec1, 0)

        melspec2 = torchaudio.transforms.MelSpectrogram(sample_rate=self.sampling_rate, n_fft=self.filter_length, win_length=self.win_length,
                                                        hop_length=self.hop_length, f_min=self.mel_fmin, f_max=self.mel_fmax,
                                                        n_mels=self.n_mel_channels)(audio1)
        melspec2 = torch.squeeze(melspec2, 0)
        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class ConcateSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(ConcateSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

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
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
