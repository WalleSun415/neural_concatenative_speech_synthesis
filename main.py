import torch
from hparams import load_hparams
from data_ulils import TextMelLoader, TextMelCollate
from torch.utils.data import DataLoader
from model import NeuralConcatenativeSpeechSynthesis
from loss_function import NeuralConcatenativeLoss
import matplotlib.pyplot as plt
import time
import math
from tensorboardX import SummaryWriter
from prefetch_generator import BackgroundGenerator
from data_ulils import get_mel_text_pair_inference
import librosa
import librosa.display
import numpy as np
from utils import dynamic_range_compression, dynamic_range_decompression


def time_since(since):
    s = time.time()-since
    m = math.floor(s / 60)
    s -= m*60
    return m, s


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate()

    train_loader = DataLoader(trainset, num_workers=2, shuffle=True,
                              sampler=None,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def validate(model, criterion, valset, batch_size, collate_fn):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(valset, sampler=None, num_workers=2,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)
        val_loss = 0.0
        total_mel_loss = 0.0
        total_gate_loss = 0.0
        for i, batch in enumerate(BackgroundGenerator(val_loader)):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss, mel_loss, gate_loss = criterion(y_pred, y)
            val_loss += loss.item()
            total_mel_loss += mel_loss
            total_gate_loss += gate_loss
        val_loss = val_loss / (i + 1)
        total_mel_loss = total_mel_loss / (i + 1)
        total_gate_loss = total_gate_loss / (i + 1)
    model.train()
    return val_loss, total_mel_loss, total_gate_loss


def train(hparams):
    torch.cuda.manual_seed(hparams.seed)

    model = NeuralConcatenativeSpeechSynthesis(hparams)
    print(model)
    print("parameter numbers: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = NeuralConcatenativeLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # move model to cuda
    model.to(device)
    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    running_loss = 0.0
    writer = SummaryWriter('runs/exp-2')
    for epoch in range(hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(BackgroundGenerator(train_loader)):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss, mel_loss, gate_loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss log and visualization
            running_loss += loss.item()
            if i%10 == 0:
                print('[%d, %d] loss: %.3f' % (epoch, i, running_loss / 10))
                running_loss = 0.0
                val_loss, total_mel_loss, total_gate_loss = validate(model, criterion, valset, hparams.batch_size, collate_fn)
                writer.add_scalar("val_loss", val_loss)
                writer.add_scalar("val_mel_loss", total_mel_loss)
                writer.add_scalar("val_gate_loss", total_gate_loss)
            writer.add_scalar("training_loss", loss.item(), i)
            writer.add_scalar("mel_loss", mel_loss, i)
            writer.add_scalar("gate_loss", gate_loss, i)
            del loss
            del y_pred
    torch.save(obj=model.state_dict(), f=hparams.model_save_path)


if __name__ == "__main__":
    hparams = load_hparams()
    torch.manual_seed(hparams.seed)
    # train(hparams)
    text = "It is an easy document to understand when you remember that it was called into being"
    inputs = get_mel_text_pair_inference(text, hparams)
    model = NeuralConcatenativeSpeechSynthesis(hparams)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('NeuralConcate.pth'))
    else:
        model.load_state_dict(torch.load('NeuralConcate.pth', map_location=torch.device('cpu')))
    with torch.no_grad():
        mel_outputs, gate_outputs = model.inference(inputs)
    print(mel_outputs.shape, gate_outputs.shape)


    y, sample_rate = librosa.core.load("/Users/swl/Dissertation/LJSpeech-1.1/wavs/LJ023-0056.wav", sr=22050)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=1024, hop_length=256, power=1,
                                                    n_mels=hparams.n_mel_channels, fmin=hparams.mel_fmin, fmax=hparams.mel_fmax)
    print(melspectrogram.shape)
    frame_num = melspectrogram.shape[1]
    mel_outputs = mel_outputs.data.numpy()[:frame_num, :].T

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(np.log(melspectrogram), y_axis='mel', x_axis='time',
                             hop_length=hparams.hop_length, fmin=hparams.mel_fmin, fmax=hparams.mel_fmax)
    plt.title('Original Mel spectrogram')
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mel_outputs, y_axis='mel', x_axis='time',
                             hop_length=hparams.hop_length, fmin=hparams.mel_fmin, fmax=hparams.mel_fmax)
    plt.title('Generated Mel spectrogram')
    plt.tight_layout()
    plt.show()