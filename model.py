import torch
from torch import nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedd_size, hidden_size,
                 num_layers=1, bidirectional=False, drop_prob=0):
        super(TextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.n_directions = 1 if not bidirectional else 2

        self.embedding = nn.Embedding(vocab_size, embedd_size)
        self.rnn = nn.GRU(embedd_size, hidden_size, num_layers,
                          bias=False, dropout=drop_prob, bidirectional=bidirectional)

    def forward(self, inputs):
        # inputs: batch * sequence_len
        batch_size = inputs.size(0)
        hidden = self._init_state(batch_size)

        embedding = self.embedding(inputs.long()).permute(1, 0, 2)
        # input: seq * batch * dim
        output, hidden = self.rnn(embedding, hidden)
        return output, hidden

    def _init_state(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return hidden


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()
        self.model = nn.Sequential(nn.Linear(dec_hidden_size, enc_hidden_size, bias=False))

    def forward(self, encoder_embeddings, encoder_states, decoder_state):
        # encoder_states = encoder_states
        e = self.model(decoder_state).unsqueeze(dim=-1)
        e = torch.matmul(encoder_states, e)  # batch * seq_len * 1
        weights = F.softmax(e, dim=1)  # batch * seq_len * 1
        return (encoder_embeddings * weights).sum(dim=0)


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedd_size, hidden_size, encoder_hidden_size,
                 num_layers=1, bidirectional=False, drop_prob=0):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.n_directions = 1 if not bidirectional else 2

        self.embedding = nn.Embedding(vocab_size, embedd_size)
        self.attention = Attention(encoder_hidden_size, hidden_size)
        self.rnn = nn.GRU(embedd_size, hidden_size, num_layers, bias=False, dropout=drop_prob, bidirectional=bidirectional)

    def forward(self, input, last_decoder_state, enc_hidden_states, enc_embeddings):
        '''
        :param inputs: (batch, )
        :param last_decoder_state: (num_layers * birectional, batch_size, hidden_size)
        :param encoder_states: (seq_len, batch_size, enc_hidden_size)
        :return: distribution, hidden_state
        '''
        # current embedding
        embedding = self.embedding(input.long()).permute(1, 0, 2)
        # current hidden state
        _, hidden = self.rnn(embedding, last_decoder_state)
        hidden = hidden.squeeze(dim=0)
        # current distribution for current hidden state
        output = self.attention(enc_embeddings, enc_hidden_states, hidden)
        return output, hidden

    def _init_state(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return hidden


if __name__ == "__main__":
    # class TextEncoder unit test
    encoder = TextEncoder(vocab_size=10, embedd_size=8, hidden_size=16)
    output, state = encoder(torch.zeros((4, 7)))
    print(output.shape, state.shape)

    # class Attention unit test
    batch_size, seq_len, enc_embedd_size, enc_num_hiddens, dec_num_hiddens = 4, 10, 32, 64, 16
    model = Attention(enc_num_hiddens, dec_num_hiddens)
    enc_embedd = torch.rand((batch_size, seq_len, enc_embedd_size))
    enc_states = torch.rand((batch_size, seq_len, enc_num_hiddens))
    dec_state = torch.randn((batch_size, dec_num_hiddens))
    print(model.forward(enc_embedd, enc_states, dec_state).shape)

    # class TextDecoder unit test
    batch_size, seq_len, dec_embedd_size, enc_num_hiddens, dec_num_hiddens = 4, 10, 12, 8, 16
    model = AttentionDecoder(vocab_size=30, embedd_size=dec_embedd_size, hidden_size=dec_num_hiddens, encoder_hidden_size=enc_num_hiddens)
    input = torch.randint(30, (batch_size, 1))
    init_state = model._init_state(batch_size)
    enc_embedd = torch.rand((batch_size, seq_len, enc_embedd_size))
    enc_states = torch.rand((batch_size, seq_len, enc_num_hiddens))
    output, hidden = model.forward(input, init_state, enc_states, enc_embedd)
    print(output.size())
    print(hidden.size())