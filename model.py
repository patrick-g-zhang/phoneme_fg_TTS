from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm, Prenet
from utils import to_gpu, get_mask_from_lengths
from modules import ReferenceEncoder

import pdb


class RefAttention(nn.Module):
    """
        reference attention which is used to align phoneme and  mel spec.
        phoneme embedding will be query and mel spec will be key and value here.
    """
    # reference  to

    def __init__(self, hp):
        super().__init__()
        self.attention_layer = Attention(
            hp.encoder_embedding_dim, hp.ref_enc_gru_size,
            hp.ref_enc_gru_size, hp.attention_location_n_filters,
            hp.attention_location_kernel_size)

    def forward(self, query, key, output_lengths):
        # query :[B, T_text, 512]
        # key: [B, T_mel, 128]
        query = query.transpose(0, 1).contiguous()  # [T_text, B, dim]
        att_outputs = []
        alignments = []
        # initilization
        self.memory = key
        B = self.memory.size(0)
        MAX_TIME = self.memory.size(1)
        # add mask
        self.processed_memory = self.attention_layer.memory_layer(
            self.memory)  # [B, T_mel, attention_dim]
        self.attention_weights = Variable(self.memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(self.memory.data.new(
            B, MAX_TIME).zero_())

        self.mask = ~get_mask_from_lengths(output_lengths)
        while len(att_outputs) < query.size(0):
            single_query = query[len(att_outputs)]  # [B, 512]
            attention_weights_cat = torch.cat(
                (self.attention_weights.unsqueeze(1),
                 self.attention_weights_cum.unsqueeze(1)), dim=1)  # [B, 2, MAX_T]
            attention_context, self.attention_weights = self.attention_layer(
                single_query, self.memory, self.processed_memory,
                attention_weights_cat, self.mask)
            self.attention_weights_cum += self.attention_weights
            att_outputs.append(attention_context)
            alignments.append(self.attention_weights)
        att_outputs = torch.stack(att_outputs).transpose(0, 1).contiguous()
        alignments = torch.stack(alignments).transpose(0, 1)
        return att_outputs, alignments


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)  # (B, 32, max_T)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')  # (B, max_T, 128)

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)  # (B, max_T, 128)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, attention_rnn_dim)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))  # (B, 1, 128)
        processed_attention_weights = self.location_layer(
            attention_weights_cat)  # (B, max_T, 128)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))  # (B, max_T, 1)

        energies = energies.squeeze(-1)   # (B, max_T)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)  # (B, max_T)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int(
                                 (hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(
                self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, spk_embeds=None):
        # x: [B, symbols_embedding_dim, T]
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)  # (B, max_len, D)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        return outputs

    def inference(self, x, spk_embeds=None):

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)  # (1, len, D)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim + hparams.ref_enc_gru_size
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.prenet_type = 'original'
        self.prenet_dropout = True
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            self.prenet_type,
            self.prenet_dropout,
            out_features=[hparams.prenet_dim, hparams.prenet_dim],
            bias=False)

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        # self.dense_init_attention_lstm = nn.Linear(
        # hparams.spks_embedding_dim, hparams.attention_rnn_dim * 2)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        # self.dense_init_decoder_lstm = nn.Linear(
        # hparams.spks_embedding_dim, hparams.decoder_rnn_dim * 2)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, spk_embeds, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        # attention_init_state = F.softsign(
        # self.dense_init_attention_lstm(spk_embeds))  # (B, 1024*2)
        # attention_init_state = attention_init_state.view(
        # -1, attention_init_state.size(1) // 2, 2)  # (B, 1024, 2)
        # attention_init_state = attention_init_state.permute(
        # 2, 0, 1)  # (2,B, 1024)
        # self.attention_hidden = attention_init_state[0]
        # self.attention_cell = attention_init_state[1]

        # one speaker case
        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        # decoder_init_state = F.softsign(
        # self.dense_init_decoder_lstm(spk_embeds))  # (B, 1024*2)
        # decoder_init_state = decoder_init_state.view(
        # -1, decoder_init_state.size(1) // 2, 2)  # (B, 1024, 2)
        # decoder_init_state = decoder_init_state.permute(2, 0, 1)  # (2,B, 1024)
        # self.decoder_hidden = decoder_init_state[0]
        # self.decoder_cell = decoder_init_state[1]

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat(
            (decoder_input, self.attention_context), -1)  # [B, 256+512]
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))        # [B, 1024]
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)  # [B, 2, MAX_T]
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)  # [B, 1024+512]
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))  # [B, 1024 ]
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths, spk_embeds=None):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        spk_embedds: speaker embedding vector 【spk_num, 16]
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(
            memory).unsqueeze(0)  # (1, B, n_mel_channels)
        decoder_inputs = self.parse_decoder_inputs(
            decoder_inputs)  # (T_out, B, n_mel_channels)
        # (1+T_out, B, n_mel_channels)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)  # (1+T_out, B, 256)

        self.initialize_decoder_states(
            memory,  mask=~get_mask_from_lengths(memory_lengths), spk_embeds=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]  # （B, 256）
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, spk_embeds=None):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None, spk_embeds=None)

        mel_outputs, gate_outputs, alignments = [], [], []

        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]
            # print(torch.sigmoid(gate_output.data))

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                # print("Warning! Reached max decoder steps")
                break
            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.ref_encoder = ReferenceEncoder(hparams)
        self.ref_attention = RefAttention(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        # text_padded, input_lengths, spk_ids, mel_padded, gate_padded, \
            # output_lengths = batch
        text_padded, input_lengths, mel_padded, \
            gate_padded, output_lengths = batch

        text_padded = to_gpu(text_padded).long()

        # spk_ids = to_gpu(spk_ids).long()
        input_lengths = to_gpu(input_lengths).long()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded,
             output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        # text_inputs: [B, T]
        text_inputs, input_lengths, mels, output_lengths = inputs
        input_lengths = input_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(
            1, 2)  # [B, symbols_embedding_dim, T]

        encoder_outputs = self.encoder(
            embedded_inputs, input_lengths)  # (B, T, phone_embedding)
        reference_outputs = self.ref_encoder(mels)  # (B, T_mel, 128)
        ref_context, ref_scores_normalized = self.ref_attention(
            encoder_outputs, reference_outputs, output_lengths)
        encoder_outputs = torch.cat((encoder_outputs, ref_context), dim=2)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=input_lengths, spk_embeds=None)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs,
                alignments, ref_scores_normalized],
            output_lengths)

    def inference_encoder(self, inputs):
        embedded_inputs = self.embedding(inputs)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        return encoder_outputs

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
