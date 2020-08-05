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


def train(hparams):
    torch.manual_seed(hparams.seed)
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
    writer = SummaryWriter('runs/exp-1')
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
            writer.add_scalar("training_loss", loss.item())
            writer.add_scalar("mel_loss", mel_loss)
            writer.add_scalar("gate_loss", gate_loss)
            del loss
            del y_pred

    torch.save(obj=model.state_dict(), f=hparams.model_save_path)

# model.load_state_dict(torch.load(hparams.model_save_path))


if __name__ == "__main__":
    hparams = load_hparams()
    train(hparams)
