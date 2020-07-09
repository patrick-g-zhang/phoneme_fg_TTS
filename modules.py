import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils import to_gpu, get_mask_from_lengths
import pdb
from torch.autograd import Variable
from layers import ConvNorm, LinearNorm, Prenet


class ReferenceEncoder(nn.Module):
    '''
    This is reference encoder for encode mel-spec 
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hp):

        super().__init__()
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(1, 1),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hp.ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(hp.n_mel_channels, 3, 1, 1, K)
        # self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
        # hidden_size=hp.ref_enc_gru_size,
        # batch_first=True)
        self.linear_out = LinearNorm(
            hp.ref_enc_filters[-1] * out_channels, hp.ref_enc_gru_size, bias=True)
        self.n_mel_channels = hp.n_mel_channels
        self.ref_enc_gru_size = hp.ref_enc_gru_size

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        # [N, 1, Ty, n_mels]
        out = inputs.view(inputs.size(0), 1, -1,
                          self.n_mel_channels)  # [N, 1, T, 80]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty, 128, n_mels]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty, 128*n_mels]

        # outs, _ = self.gru(out)
        outs = self.linear_out(out)
        return outs

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class AttentionLayer(nn.Module):
    '''
            Attention layer according to https://arxiv.org/abs/1409.0473.
            reference code : https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/models/modules/attention.py
            Params:
                num_units: Number of units used in the attention layer
    '''

    def __init__(self, hp, normalize=True, bias=True, output_transform=False, output_nonlinearity='tanh', output_size=None):
        super(AttentionLayer, self).__init__()
        self.mode = hp.mode

        self.query_size = hp.syl_embedding_dim
        self.key_size = hp.ref_enc_gru_size
        self.value_size = hp.ref_enc_gru_size
        self.normalize = normalize
        self.softmax = nn.Softmax(dim=2)

        def wn_func(x): return x
        self.scale = nn.Parameter(torch.Tensor([1]))
        if output_transform:
            output_size = output_size or query_size
            self.linear_out = wn_func(
                nn.Linear(query_size + value_size, output_size, bias=bias))
            self.output_size = output_size
        else:
            self.output_size = self.value_size
        # if query_transform:
        self.linear_q = wn_func(
            nn.Linear(self.query_size, self.key_size, bias=bias))
        self.dropout = nn.Dropout(hp.rf_att_dropout)
        self.output_nonlinearity = output_nonlinearity
        self.mask = None
        self.score_mask_value = -float("inf")

    def set_mask(self, mask):
        # applies a mask of b x t length
        self.mask = mask
        if mask is not None and not self.batch_first:
            self.mask = self.mask.t()

    def calc_score(self, att_query, att_keys):
        """
        att_query is: b x t_q x n
        att_keys is b x t_k x n
        return b x t_q x t_k scores
        """

        b, t_k, n = list(att_keys.size())
        t_q = att_query.size(1)
        if self.mode == 'bahdanau':
            att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
            att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
            sum_qk = att_query + att_keys
            sum_qk = sum_qk.view(b * t_k * t_q, n)
            out = self.linear_att(F.tanh(sum_qk)).view(b, t_q, t_k)
        elif self.mode == 'dot_prod':
            out = torch.bmm(att_query, att_keys.transpose(1, 2))
            if hasattr(self, 'scale'):
                out = out * self.scale
        return out

    def forward(self, query, keys, output_lengths):
        """
            query:  b x t_q x n
            keys: b x t_k x n
        """

        values = keys

        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)
        # Fully connected layers to transform query
        att_query = self.linear_q(query)

        scores = self.calc_score(att_query, keys)  # size b x t_q x t_k

        # add mask
        mask = ~get_mask_from_lengths(output_lengths)
        mask = mask.expand(t_q, mask.size(0), mask.size(1)).permute(1, 0, 2)
        scores.masked_fill_(mask, self.score_mask_value)

        # Normalize the scores
        scores_normalized = F.softmax(scores, dim=2)  # size t_q x t_k

        # Calculate the weighted average of the attention inputs
        # according to the scores
        scores_normalized = self.dropout(scores_normalized)
        context = torch.bmm(scores_normalized, values)  # b x t_q x n

        if hasattr(self, 'linear_out'):
            context = self.linear_out(torch.cat([query, context], 2))
            if self.output_nonlinearity == 'tanh':
                context = F.tanh(context)
            elif self.output_nonlinearity == 'relu':
                context = F.relu(context, inplace=True)

        return context, scores_normalized
