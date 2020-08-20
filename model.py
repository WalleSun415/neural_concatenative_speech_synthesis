import torch
from torch import nn
import torch.nn.functional as F
from utils import to_gpu, get_mask_from_lengths
from layers import LinearNorm, ConvNorm
import random

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, hparams):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

        self.convolutions = nn.Sequential(
                ConvNorm(hparams.prenet_dim,
                         hparams.prenet_dim,
                         kernel_size=hparams.audio_kernel_size, stride=hparams.audio_stride, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.prenet_dim))

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0, training=self.training)
        x = x.permute(1, 2, 0)
        x = self.convolutions(x).permute(2, 0, 1)
        return F.dropout(F.relu(x), p=0, training=self.training)


class TargetPrenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(TargetPrenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0, training=self.training)
        return x


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
        inputs = self.embedding(inputs.long()).permute(1, 0, 2)
        # input: seq * batch * dim
        output, hidden = self.rnn(inputs)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()
        self.model = LinearNorm(dec_hidden_size, enc_hidden_size, bias=False)

    def forward(self, encoder_inputs, encoder_states, decoder_states):
        # seq_len, batch, hidden_size
        decoder_states = self.model(decoder_states).permute(1, 0, 2)
        encoder_states = encoder_states.permute(1, 2, 0)
        decoder_states = torch.matmul(decoder_states, encoder_states)
        encoder_inputs = encoder_inputs.permute(1, 0, 2)
        decoder_states = F.softmax(decoder_states, dim=2)  # batch * seq_len * 1
        return torch.matmul(decoder_states, encoder_inputs).permute(1, 0, 2)


class AttentionLoop(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, method="concat"):
        super(AttentionLoop, self).__init__()
        self.method = method
        if method == "general":
            self.model = LinearNorm(dec_hidden_size, enc_hidden_size, bias=False)
        elif method == "concat":
            self.fc = LinearNorm(dec_hidden_size, enc_hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(enc_hidden_size, 1))

    def forward(self, encoder_states, decoder_state):
        if self.method == "general":
            # seq_len, batch, hidden_size
            encoder_states = encoder_states.permute(1, 0, 2)
            decoder_state = self.model(decoder_state).unsqueeze(dim=-1)
            decoder_state = torch.matmul(encoder_states, decoder_state)  # batch * seq_len * 1
            decoder_state = F.softmax(decoder_state, dim=1)  # batch * seq_len * 1
            return (encoder_states * decoder_state).sum(dim=1)
        elif self.method == "concat":
            out = torch.tanh(self.fc(decoder_state + encoder_states))
            out = torch.matmul(out, self.weight).permute(1, 0, 2)
            out = F.softmax(out, dim=1)
            return (encoder_states.permute(1, 0, 2) * out).sum(dim=1)


class AttentionDecoder(nn.Module):
    def __init__(self, embedd_size, decoder_hidden_size, encoder_hidden_size,
                 num_layers=1, bidirectional=False, drop_prob=0):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = decoder_hidden_size
        self.n_layers = num_layers
        self.n_directions = 1 if not bidirectional else 2
        self.rnn = nn.GRU(embedd_size, decoder_hidden_size, num_layers, bias=False,
                          dropout=drop_prob, bidirectional=bidirectional)
        self.attention = Attention(encoder_hidden_size, decoder_hidden_size)

    def forward(self, decoder_inputs, enc_hidden_states, encoder_inputs):
        '''
        :param inputs: (batch, seq_len, hidden_size)
        :param last_decoder_state: (num_layers * birectional, batch_size, hidden_size)
        :param encoder_states: (seq_len, batch_size, enc_hidden_size)
        :return: distribution, hidden_state
        '''
        # sequence of hidden state
        decoder_hidden_states, _ = self.rnn(decoder_inputs)
        attention_distribution = self.attention(encoder_inputs, enc_hidden_states, decoder_hidden_states)
        return decoder_hidden_states, attention_distribution


class AudioEncoder(nn.Module):
    def __init__(self, spectral_features_size, hidden_size,
                 num_layers=1, bidirectional=False, drop_prob=0):
        super(AudioEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.n_directions = 1 if not bidirectional else 2
        self.rnn = nn.GRU(spectral_features_size, hidden_size, num_layers,
                          bias=False, dropout=drop_prob, bidirectional=bidirectional)

    def forward(self, inputs):
        # inputs: (batch, sequence_len)
        output, hidden = self.rnn(inputs)
        return output, hidden


class RecurrentDecoder(nn.Module):
    def __init__(self, input_size, decoder_hidden_size, audio_encoder_size, spectral_size, hparams,
                 num_layers=1, bidirectional=False, drop_prob=0):
        super(RecurrentDecoder, self).__init__()
        self.batch_size = hparams.batch_size
        self.decoder_hidden_size = decoder_hidden_size
        self.rnn_dropout = hparams.rnn_dropout
        self.n_mel_channels = hparams.n_mel_channels

        self.attention = AttentionLoop(audio_encoder_size, decoder_hidden_size, method="concat")
        self.rnn = nn.GRUCell(input_size+audio_encoder_size, decoder_hidden_size, bias=False)
        self.spectral_linear_projection = LinearNorm(audio_encoder_size, audio_encoder_size)
        self.gate_linear_projection = LinearNorm(audio_encoder_size, 1, bias=True, w_init_gain='sigmoid')

    def decode(self, decoder_input):
        # self.decoder_current_state = self.rnn(decoder_input, self.decoder_current_state)
        # attention_context = self.attention(self.alignment_inputs, self.decoder_current_state)
        # decoder_hidden_attention_context = torch.cat((self.decoder_current_state, attention_context), dim=-1)
        # mel_output = self.spectral_linear_projection(decoder_hidden_attention_context)
        # gate_output = self.gate_linear_projection(decoder_hidden_attention_context)

        attention_context = self.attention(self.alignment_inputs, self.decoder_current_state)
        input_and_context = torch.cat((decoder_input, attention_context), dim=-1)
        self.decoder_current_state = self.rnn(input_and_context, self.decoder_current_state)
        # self.decoder_current_state = F.relu(F.dropout(self.decoder_current_state, self.rnn_dropout, self.training))
        # decoder_hidden_attention_context = torch.cat((attention_context, self.decoder_current_state), dim=-1)
        mel_output = self.spectral_linear_projection(self.decoder_current_state)
        gate_output = self.gate_linear_projection(self.decoder_current_state)
        return mel_output, gate_output

    def parse_decoder_outputs(self, mel_outputs, gate_outputs):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        """
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        return mel_outputs, gate_outputs

    def forward(self, decoder_inputs, alignment_inputs):
        '''
        :param decoder_inputs: target mel spectral features -> (batch, n_mel_channel, seq_len)
        :param alignment_inputs: "glued" mel spectral features
        :return:
        '''
        batch_size = decoder_inputs.size(1)
        self.decoder_current_state = torch.zeros((batch_size, self.decoder_hidden_size), requires_grad=True)
        self.decoder_current_state = to_gpu(self.decoder_current_state)
        init_state = self.init_state(alignment_inputs).unsqueeze(0)
        init_state = to_gpu(init_state).float()
        decoder_inputs = torch.cat((init_state, decoder_inputs), dim=0)
        mel_outputs, gate_outputs = [], []
        # decoder_input = self.alignment_inputs[-1, :, :]
        decoder_input = decoder_inputs[len(mel_outputs)]
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            mel_output, gate_output = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            decoder_input = decoder_inputs[len(mel_outputs)]
            # if teacher_forcing:
            #    decoder_input = decoder_inputs[len(mel_outputs)]
            # else:
            #     decoder_input = mel_output

        mel_outputs, gate_outputs = self.parse_decoder_outputs(mel_outputs, gate_outputs)
        return mel_outputs, gate_outputs

    def inference(self, alignment_inputs):
        self.decoder_current_state = torch.zeros((1, self.decoder_hidden_size), requires_grad=True)
        self.decoder_current_state = to_gpu(self.decoder_current_state)
        decoder_input = self.init_state(alignment_inputs)
        decoder_input = to_gpu(decoder_input).float()
        mel_outputs, gate_outputs = [], []
        # decoder_input = self.alignment_inputs[-1, :, :]
        while True:
            mel_output, gate_output = self.decode(decoder_input)
            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            threshold = torch.sigmoid(gate_output.data)
            # print("stop threshold: ", threshold)
            if threshold > 0.5:
                print("Stop threshold: ", threshold)
                print("Frame number: ", len(mel_outputs))
                break
            elif len(mel_outputs) == 1000:
                print("Warning! Reached max decoder steps. stop threshold: ", threshold)
                break
            decoder_input = mel_output
        mel_outputs, gate_outputs = self.parse_decoder_outputs(mel_outputs, gate_outputs)
        return mel_outputs, gate_outputs

    def init_state(self, alignment_inputs):
        # decoder_inputs: (batch, n_mel_channel, seq_len)
        batch_size = alignment_inputs.size(1)
        hidden_size = alignment_inputs.size(2)
        decoder_input = torch.zeros((batch_size, hidden_size), requires_grad=False)
        self.alignment_inputs = alignment_inputs
        return decoder_input


class NeuralConcatenativeSpeechSynthesis(nn.Module):
    def __init__(self, hparams):
        super(NeuralConcatenativeSpeechSynthesis, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.audio_prenet = Prenet(
            hparams.n_mel_channels,
            [hparams.prenet_dim, hparams.prenet_dim], hparams)
        self.target_audio_prenet = TargetPrenet(
            hparams.n_mel_channels,
            [hparams.prenet_dim, hparams.prenet_dim])
        self.text_prenet = ConvNorm(hparams.symbols_embedding_dim, hparams.symbols_embedding_dim,
                                    kernel_size=hparams.decoder_kernel_size, stride=hparams.text_stride)

        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        # Text to audio seq2seq(alignment 1 module)
        self.glued_mel_encoder = AudioEncoder(hparams.prenet_dim, hparams.encoder_rnn_dim)
        self.glued_text_decoder = AttentionDecoder(hparams.symbols_embedding_dim,
                                                   hparams.decoder_rnn_dim, hparams.encoder_rnn_dim)
        # Text to text seq2seq(Pseudo alignment 2)
        self.target_text_decoder = AttentionDecoder(hparams.symbols_embedding_dim,
                                                    hparams.decoder_rnn_dim, hparams.decoder_rnn_dim)
        # Decoder
        self.decoder = RecurrentDecoder(hparams.prenet_dim, hparams.mel_decoder_rnn_dim,
                                        hparams.prenet_dim, hparams.n_mel_channels, hparams)
        self.postnet = LinearNorm(hparams.prenet_dim, hparams.n_mel_channels)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, \
        glued_text_padded, glued_mel_padded = batch

        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        glued_text_padded = to_gpu(glued_text_padded).long()
        glued_mel_padded = to_gpu(glued_mel_padded).float()
        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, glued_text_padded, glued_mel_padded),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask[:, 0, :], 10)  # gate energies

        return outputs

    def forward(self, inputs):
        text_padded, text_lengths, mel_padded, max_len, output_lengths, glued_text_padded, glued_mel_padded = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        # Text to audio seq2seq(alignment 1 module)
        glued_mel_padded = glued_mel_padded.permute(2, 0, 1)
        glued_mel_padded = self.audio_prenet(glued_mel_padded)
        glued_audio_encoder_output, _ = self.glued_mel_encoder(glued_mel_padded)

        glued_text_padded = self.embedding(glued_text_padded).permute(0, 2, 1)
        glued_text_padded = F.dropout(F.relu(self.text_prenet(glued_text_padded)), p=0.5, training=self.training).permute(2, 0, 1)
        glued_text_hidden_states, alignment_input = self.glued_text_decoder(glued_text_padded,
                                                                            glued_audio_encoder_output,
                                                                            glued_mel_padded)
        del glued_text_padded
        del glued_mel_padded
        del glued_audio_encoder_output
        # Text to text seq2seq(Pseudo alignment 2)
        text_padded = self.embedding(text_padded).permute(1, 0, 2)
        _, weighted_alignment = self.target_text_decoder(text_padded, glued_text_hidden_states, alignment_input)
        del text_padded
        # Decoder
        mel_padded = mel_padded.permute(2, 0, 1)
        mel_padded = self.target_audio_prenet(mel_padded)
        mel_outputs, gate_outputs = self.decoder(mel_padded, weighted_alignment)
        mel_outputs = self.postnet(F.relu(mel_outputs)).transpose(1, 2)
        del mel_padded
        del weighted_alignment
        return self.parse_output(
            [mel_outputs, gate_outputs],
            output_lengths)

    def inference(self, inputs):
        text_padded, glued_text_padded, glued_mel_padded = inputs

        text_padded = to_gpu(text_padded).long()
        glued_text_padded = to_gpu(glued_text_padded).long()
        glued_mel_padded = to_gpu(glued_mel_padded).float()

        # Text to audio seq2seq(alignment 1 module)
        glued_mel_padded = glued_mel_padded.permute(2, 0, 1)
        glued_mel_padded = self.audio_prenet(glued_mel_padded)
        glued_audio_encoder_output, _ = self.glued_mel_encoder(glued_mel_padded)

        glued_text_padded = self.embedding(glued_text_padded).permute(0, 2, 1)
        glued_text_padded = F.relu(self.text_prenet(glued_text_padded)).permute(2, 0, 1)
        glued_text_hidden_states, alignment_input = self.glued_text_decoder(glued_text_padded,
                                                                            glued_audio_encoder_output,
                                                                            glued_mel_padded)

        text_padded = self.embedding(text_padded).permute(1, 0, 2)
        _, weighted_alignment = self.target_text_decoder(text_padded,
                                                         glued_text_hidden_states, alignment_input)
        mel_outputs, gate_outputs = self.decoder.inference(weighted_alignment)
        mel_outputs = self.postnet(F.relu(mel_outputs))
        return mel_outputs.squeeze(0), gate_outputs


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
    # print(model.forward(enc_embedd, enc_states, dec_state).shape)

    # class TextDecoder unit test
    batch_size, seq_len, dec_embedd_size, enc_num_hiddens, dec_num_hiddens = 4, 10, 12, 64, 16
    model = AttentionDecoder(vocab_size=30, embedd_size=dec_embedd_size,
                        decoder_hidden_size=dec_num_hiddens, encoder_hidden_size=enc_num_hiddens)
    print(model.forward(torch.zeros((4, 7)), enc_states, enc_embedd))
