# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
from config import global_config as cfg
import copy, random

from reader import pad_sequences


def cuda_(var):
    return var.cuda() if cfg.cuda else var


def toss_(p):
    return random.randint(0, 99) <= p


def nan(v):
    if type(v) is float:
        return v == float('nan')
    return np.isnan(np.sum(v.data.cpu().numpy()))


def get_sparse_input_aug(x_input_np):
    """
    sparse input of
    :param x_input_np: [T,B]
    :return: Numpy array: [B,T,aug_V]
    """
    ignore_index = [0]
    unk = 2
    result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size + x_input_np.shape[0]),
                      dtype=np.float32)
    result.fill(1e-10)
    for t in range(x_input_np.shape[0]):
        for b in range(x_input_np.shape[1]):
            w = x_input_np[t][b]
            if w not in ignore_index:
                if w != unk:
                    result[t][b][x_input_np[t][b]] = 1.0
                else:
                    result[t][b][cfg.vocab_size + t] = 1.0
    result_np = result.transpose((1, 0, 2))
    result = torch.from_numpy(result_np).float()
    return result


def init_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal(hh[i:i + gru.hidden_size], gain=1)


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.zeros(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, normalize=True):
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,H]
        attn_energies = self.score(hidden, encoder_outputs)
        normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]
        context = torch.bmm(normalized_energy, encoder_outputs)  # [B,1,H]
        return context.transpose(0, 1)  # [1,B,H]

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy


class SimpleDynamicEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.emb_ctrl = nn.Linear(embed_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        init_gru(self.gru)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        ctrl_embedded = self.emb_ctrl(embedded)
        embedded = embedded + ctrl_embedded
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden, embedded


class BinaryClassification(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate):
        super().__init__()

        # [change] GRU was replaced by a linear ffnn with one hidden layer, the content of the hidden layer 
        # is used for later attentions instead of last GRU hidden state
        self.ffnn_hidden = nn.Linear(hidden_size + embed_size, hidden_size)
        self.ffnn_out = nn.Linear(hidden_size, 1)
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.emb_ctrl = nn.Linear(embed_size, embed_size)
        self.dropout_rate = dropout_rate
        self.attn_u = Attn(hidden_size)

    def forward(self, u_enc_out, z_tm1, last_hidden):
        context = self.attn_u(last_hidden, u_enc_out)
        embed_z = self.emb(z_tm1)
        ctrl_embed_z = self.emb_ctrl(embed_z)
        embed_z = ctrl_embed_z + embed_z
        embed_z = F.dropout(embed_z, self.dropout_rate)
        ffnn_in = torch.cat([embed_z, context], 2)
        last_hidden = self.ffnn_hidden(ffnn_in)
        logit = self.ffnn_out(last_hidden).squeeze(0)
        return logit, last_hidden


class SlotBinaryClassification(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, degree_size, dropout_rate):
        super().__init__()

        # [change] GRU was replaced by a linear ffnn with one hidden layer, the content of the hidden layer 
        # is used for later attentions instead of last GRU hidden state
        self.ffnn_hidden = nn.Linear(hidden_size + embed_size + degree_size, hidden_size)
        self.ffnn_out = nn.Linear(hidden_size, 1)
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.emb_ctrl = nn.Linear(embed_size, embed_size)
        self.attn_u = Attn(hidden_size)
        self.dropout_rate = dropout_rate

    def forward(self, u_enc_out, z_tm1, last_hidden, degree_input):
        context = self.attn_u(last_hidden, u_enc_out)
        embed_z = self.emb(z_tm1)
        ctrl_embed_z = self.emb_ctrl(embed_z)
        embed_z = ctrl_embed_z + embed_z
        embed_z = F.dropout(embed_z, self.dropout_rate)
        ffnn_in = torch.cat([embed_z, context, degree_input], 2)
        last_hidden = self.ffnn_hidden(ffnn_in)
        logit = self.ffnn_out(last_hidden).squeeze(0)
        return logit, last_hidden

class BSpanDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate):
        super().__init__()
        self.ffnn_hidden = nn.Linear(hidden_size * 2 + embed_size, hidden_size)
        self.ffnn_out = nn.Linear(hidden_size, vocab_size)
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.emb_ctrl = nn.Linear(embed_size, embed_size)
        self.attn_u = Attn(hidden_size)
        self.proj_query = nn.Linear(hidden_size + embed_size, hidden_size)
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_rate = dropout_rate

    def forward(self, u_enc_out, z_tm1, last_hidden, u_input_np, pv_z_enc_out, prev_z_input_np, u_emb, pv_z_emb):

        sparse_u_input = Variable(get_sparse_input_aug(u_input_np), requires_grad=False)

        if pv_z_enc_out is not None:
            context = self.attn_u(last_hidden, torch.cat([pv_z_enc_out, u_enc_out], dim=0))
        else:
            context = self.attn_u(last_hidden, u_enc_out)

        embed_z = self.emb(z_tm1)
        ctrl_embed_z = self.emb_ctrl(embed_z)
        embed_z = ctrl_embed_z + embed_z
        embed_z = F.dropout(embed_z, self.dropout_rate)
        '''
        query = self.proj_query(torch.cat([embed_z, last_hidden], dim=2))
        if pv_z_enc_out is not None:
            context = self.attn_u(query, torch.cat([pv_z_enc_out, u_enc_out], dim=0))
        else:
            context = self.attn_u(query, u_enc_out)
        '''

        # [change] GRU was replaced by a linear ffnn with one hidden layer, the content of the hidden layer 
        # is used for later attentions instead of GRU hidden states
        ffnn_in = torch.cat([embed_z, context, last_hidden], 2)
        last_hidden = self.ffnn_hidden(ffnn_in)
        gen_score = self.ffnn_out(last_hidden).squeeze(0)
        gen_score = F.dropout(gen_score, self.dropout_rate)
        u_copy_score = torch.tanh(self.proj_copy1(u_enc_out.transpose(0, 1)))  # [B,T,H]
        # stable version of copynet
        u_copy_score = torch.matmul(u_copy_score, last_hidden.squeeze(0).unsqueeze(2)).squeeze(2)
        u_copy_score = u_copy_score.cpu()
        u_copy_score_max = torch.max(u_copy_score, dim=1, keepdim=True)[0]
        u_copy_score = torch.exp(u_copy_score - u_copy_score_max)  # [B,T]
        u_copy_score = torch.log(torch.bmm(u_copy_score.unsqueeze(1), sparse_u_input)).squeeze(
            1) + u_copy_score_max  # [B,V]
        u_copy_score = cuda_(u_copy_score)
        if pv_z_enc_out is None:
            u_copy_score = F.dropout(u_copy_score, self.dropout_rate)
            scores = F.softmax(torch.cat([gen_score, u_copy_score], dim=1), dim=1)
            gen_score, u_copy_score = scores[:, :cfg.vocab_size], \
                                      scores[:, cfg.vocab_size:]
            proba = gen_score + u_copy_score[:, :cfg.vocab_size]  # [B,V]
            proba = torch.cat([proba, u_copy_score[:, cfg.vocab_size:]], 1)
        else:
            sparse_pv_z_input = Variable(get_sparse_input_aug(prev_z_input_np), requires_grad=False)
            pv_z_copy_score = torch.tanh(self.proj_copy2(pv_z_enc_out.transpose(0, 1)))  # [B,T,H]
            pv_z_copy_score = torch.matmul(pv_z_copy_score, last_hidden.squeeze(0).unsqueeze(2)).squeeze(2)
            pv_z_copy_score = pv_z_copy_score.cpu()
            pv_z_copy_score_max = torch.max(pv_z_copy_score, dim=1, keepdim=True)[0]
            pv_z_copy_score = torch.exp(pv_z_copy_score - pv_z_copy_score_max)  # [B,T]
            pv_z_copy_score = torch.log(torch.bmm(pv_z_copy_score.unsqueeze(1), sparse_pv_z_input)).squeeze(
                1) + pv_z_copy_score_max  # [B,V]
            pv_z_copy_score = cuda_(pv_z_copy_score)
            scores = F.softmax(torch.cat([gen_score, u_copy_score, pv_z_copy_score], dim=1), dim=1)
            gen_score, u_copy_score, pv_z_copy_score = scores[:, :cfg.vocab_size], \
                                                       scores[:,
                                                       cfg.vocab_size:2 * cfg.vocab_size + u_input_np.shape[0]], \
                                                       scores[:, 2 * cfg.vocab_size + u_input_np.shape[0]:]
            proba = gen_score + u_copy_score[:, :cfg.vocab_size] + pv_z_copy_score[:, :cfg.vocab_size]  # [B,V]
            proba = torch.cat([proba, pv_z_copy_score[:, cfg.vocab_size:], u_copy_score[:, cfg.vocab_size:]], 1)

        return last_hidden, proba


class ResponseDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, degree_size, dropout_rate, gru, proj, emb, vocab, num_head):
        super().__init__()
        self.emb = emb
        self.emb_ctrl = nn.Linear(embed_size, embed_size)
        self.attn_z = Attn(hidden_size)
        self.attn_u = Attn(hidden_size)
        self.gru = gru
        init_gru(self.gru)
        self.proj = proj
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_rate = dropout_rate

        self.vocab = vocab

    def get_sparse_selective_input(self, x_input_np, x_proba):
        result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size + x_input_np.shape[0]),
                          dtype=np.float32)
        result.fill(1e-10)
        # reqs = ['address', 'phone', 'postcode', 'pricerange', 'area']
        for t in range(x_input_np.shape[0]):
            for b in range(x_input_np.shape[1]):
                w = x_input_np[t][b]
                word = self.vocab.decode(w)
                if w == 2 or w >= cfg.vocab_size:
                    result[t][b][cfg.vocab_size + t] = 5.0
                else:
                    # if word not in ['EOS_food', 'EOS_area', 'EOS_pricerange', 'EOS_U']:
                    if 'EOS_' not in word and w != 0:
                        result[t][b][w] = result[t][b][w] + 1.0 * x_proba[t][b]

        result_np = result.transpose((1, 0, 2))
        result = torch.from_numpy(result_np).float()
        return result

    def forward(self, z_enc_out, u_enc_out, u_input_np, m_t_input, degree_input, last_hidden, z_input_np, z_proba_np):
        sparse_z_input = Variable(self.get_sparse_selective_input(z_input_np, z_proba_np),
                                  requires_grad=False)  #singal encoded sentence
        m_embed = self.emb(m_t_input)
        ctrl_m_embed = self.emb_ctrl(m_embed)
        m_embed = m_embed + ctrl_m_embed
        z_context = self.attn_z(last_hidden, z_enc_out)

        u_context = self.attn_u(last_hidden, u_enc_out)
        gru_in = torch.cat([m_embed, u_context, z_context, degree_input.unsqueeze(0)], dim=2)
        gru_out, last_hidden = self.gru(gru_in, last_hidden)
        gen_score = self.proj(torch.cat([z_context, u_context, gru_out], 2)).squeeze(0)

        z_copy_score = torch.tanh(self.proj_copy2(z_enc_out.transpose(0, 1)))
        z_copy_score = torch.matmul(z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
        z_copy_score = z_copy_score.cpu()
        z_copy_score_max = torch.max(z_copy_score, dim=1, keepdim=True)[0]
        z_copy_score = torch.exp(z_copy_score - z_copy_score_max)  # [B,T]
        z_copy_score = torch.log(torch.bmm(z_copy_score.unsqueeze(1), sparse_z_input)).squeeze(
            1) + z_copy_score_max  # [B,V]
        z_copy_score = cuda_(z_copy_score)

        scores = F.softmax(torch.cat([gen_score, z_copy_score], dim=1), dim=1)
        gen_score, z_copy_score = scores[:, :cfg.vocab_size], \
                                  scores[:, cfg.vocab_size:]
        proba = gen_score + z_copy_score[:, :cfg.vocab_size]  # [B,V]
        proba = torch.cat([proba, z_copy_score[:, cfg.vocab_size:]], 1)
        return proba, last_hidden, gru_out


class FSDM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, degree_size, layer_num, dropout_rate, z_length,
                 max_ts, num_head, separate_enc, beam_search=False, teacher_force=100, **kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']
        self.separate_enc = separate_enc
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.dec_gru = nn.GRU(degree_size + embed_size + hidden_size * 2, hidden_size,
                              dropout=dropout_rate)

        self.proj = nn.Linear(hidden_size * 3, vocab_size)
        if separate_enc:
            self.z_encoder = SimpleDynamicEncoder(vocab_size, embed_size, hidden_size, layer_num, dropout_rate)
            self.enc_w = torch.nn.parameter.Parameter(torch.Tensor(2))
            stdv = 1. / math.sqrt(self.enc_w.size(0))
            self.enc_w.data.uniform_(-stdv, stdv)
            
        self.u_encoder = SimpleDynamicEncoder(vocab_size, embed_size, hidden_size, layer_num, dropout_rate)
        self.z_decoders = nn.ModuleList(
                [BSpanDecoder(embed_size, hidden_size, vocab_size, dropout_rate) for _ in range(num_head)])
        self.req_classifiers = BinaryClassification(embed_size, hidden_size, vocab_size, dropout_rate)
        self.res_classifiers = SlotBinaryClassification(embed_size, hidden_size, vocab_size, cfg.degree_size,
                                                        dropout_rate)
        self.m_decoder = ResponseDecoder(embed_size, hidden_size, vocab_size, degree_size, dropout_rate,
                                         self.dec_gru, self.proj, self.emb, self.vocab, 1)
        self.embed_size = embed_size
        self.num_head = num_head
        self.z_length = z_length
        self.max_ts = max_ts
        self.beam_search = beam_search
        self.teacher_force = teacher_force

        self.pr_loss = nn.NLLLoss(ignore_index=0)  #reduction='elementwise_mean'
        self.dec_loss = nn.NLLLoss(ignore_index=0)
        self.multilabel_loss = nn.MultiLabelSoftMarginLoss()  #elementwise_mean’ | ‘sum’

        self.saved_log_policy = []

        if self.beam_search:
            self.beam_size = kwargs['beam_size']
            self.eos_token_idx = kwargs['eos_token_idx']

    def forward(self, u_input, u_input_np, m_input, m_input_np, z_input, u_len, m_len, turn_states,
                degree_input, k_input, i_input, r_input, mode, database=None, reader=None, constraint_eos=None,
                loss_weights=[1., 1., 1., 1.],
                **kwargs):
        requested_7 = kwargs.get('requested_7').squeeze(2).transpose(1, 0)
        response_7 = kwargs.get('response_7').squeeze(2).transpose(1, 0)
        if mode == 'train' or mode == 'valid':
            pz_proba, pm_dec_proba, turn_states, p_requested, p_response = \
                self.forward_turn(u_input, u_len, m_input=m_input, m_len=m_len, z_input=z_input, k_input=k_input,
                                  i_input=i_input, r_input=r_input, mode='train',
                                  turn_states=turn_states, degree_input=degree_input, u_input_np=u_input_np,
                                  m_input_np=m_input_np, **kwargs)
            loss, pr_loss, m_loss, requested_7_loss, response_7_loss = self.supervised_loss(torch.log(pz_proba),
                                                                                            torch.log(pm_dec_proba),
                                                                                            z_input, m_input,
                                                                                            p_requested, p_response,
                                                                                            requested_7, response_7,
                                                                                            loss_weights)
            return loss, pr_loss, m_loss, turn_states, requested_7_loss, response_7_loss

        elif mode == 'test':
            m_output_index, pz_index, turn_states = self.forward_turn(u_input, u_len=u_len, mode='test',
                                                                      turn_states=turn_states,
                                                                      degree_input=degree_input,
                                                                      u_input_np=u_input_np, m_input_np=m_input_np,
                                                                      database=database, reader=reader, k_input=k_input,
                                                                      i_input=i_input, r_input=r_input,
                                                                      constraint_eos=constraint_eos,
                                                                      **kwargs,
                                                                      )
            return m_output_index, pz_index, turn_states
        elif mode == 'rl':
            loss = self.forward_turn(u_input, u_len=u_len, is_train=False, mode='rl',
                                     turn_states=turn_states,
                                     degree_input=degree_input,
                                     u_input_np=u_input_np, m_input_np=m_input_np,
                                     **kwargs
                                     )
            return loss

    def forward_turn(self, u_input, u_len, turn_states, mode, degree_input, u_input_np, m_input_np=None,
                     m_input=None, m_len=None, z_input=None, k_input=None, i_input=None, r_input=None, database=None,
                     reader=None, constraint_eos=None, z_input_np=None, **kwargs):
        """
        compute required outputs for a single dialogue turn. Turn state{Dict} will be updated in each call.
        :param u_input_np:
        :param m_input_np:
        :param u_len:
        :param turn_states:
        :param is_train:
        :param u_input: [T,B]
        :param m_input: [T,B]
        :param z_input: [T,B]
        :return:
        """
        prev_z_input = kwargs.get('prev_z_input', None)
        prev_z_input_np = kwargs.get('prev_z_input_np', None)
        prev_z_len = kwargs.get('prev_z_len', None)
        pv_z_emb = None
        batch_size = u_input.size(1)
        pv_z_enc_out = None

        if prev_z_input is not None:
            if self.separate_enc:
                pv_z_enc_out, pv_z_hidden, pv_z_emb = self.z_encoder(prev_z_input, prev_z_len)
            else:
                pv_z_enc_out, pv_z_hidden, pv_z_emb = self.u_encoder(prev_z_input, prev_z_len)
        u_enc_out, u_enc_hidden, u_emb = self.u_encoder(u_input, u_len)
        if self.separate_enc and prev_z_input is not None:
            softmax_enc_w = torch.nn.functional.softmax(self.enc_w, dim=-1)
            last_hidden = softmax_enc_w[0]*u_enc_hidden[:-1] + softmax_enc_w[1]*pv_z_hidden[:-1]
        else:
            last_hidden = u_enc_hidden[:-1]

        m_tm1 = cuda_(Variable(torch.ones(1, batch_size).long()))  # GO token
        requestable_key = kwargs.get('requestable_key')
        requestable_slot = kwargs.get('requestable_slot')
        requestable_key_np = kwargs.get('requestable_key_np')
        requestable_slot_np = kwargs.get('requestable_slot_np')
        if mode == 'train':
            pz_dec_outs = []
            pz_proba = []
            z_length = z_input.size(0) if z_input is not None else self.z_length  # GO token
            r_length = r_input.size(0)
            hiddens = [None] * batch_size
            i_proba = []
            i_dec_outs = []
            for k_index, k_batch in enumerate(k_input):  # batch of one key
                i_length = i_input.size(1)
                k_i_input = i_input[k_index]
                k_tm1 = k_batch.unsqueeze(0)
                k_proba = []
                k_dec_outs = []
                k_last_hidden = last_hidden
                if self.num_head == 1:
                    z_idx = -1
                else:
                    z_idx = k_index
                for t in range(i_length):
                    k_last_hidden, proba = \
                            self.z_decoders[z_idx](u_enc_out=u_enc_out, u_input_np=u_input_np,  # self.z_decoders[k_index]
                                                   z_tm1=k_tm1, last_hidden=k_last_hidden,
                                                   pv_z_enc_out=pv_z_enc_out, prev_z_input_np=prev_z_input_np,
                                                   u_emb=u_emb, pv_z_emb=pv_z_emb)

                    k_proba.append(proba.unsqueeze(0))
                    k_dec_outs.append(k_last_hidden)
                    k_tm1 = k_i_input[t].unsqueeze(0)
                i_proba.append(torch.cat(k_proba, dim=0))
                i_dec_outs.append(torch.cat(k_dec_outs, dim=0))

            concat_i_dec_outs = torch.cat(i_dec_outs, dim=0)
            concat_i_proba = torch.cat(i_proba, dim=0)

            pz_dec_outs = torch.cat(i_dec_outs, dim=0)  # [Tz,B,H]
            pz_proba = concat_i_proba
            #
            #
            req_hiddens = []
            req_logits = []

            for req_idx, req_key in enumerate(requestable_key):
                req_tm = req_key.unsqueeze(0)
                if prev_z_input is not None:
                    req_logit, req_hidden = self.req_classifiers(u_enc_out=torch.cat([pv_z_enc_out, u_enc_out], dim=0), last_hidden=last_hidden,
                                                             z_tm1=req_tm)
                else:
                    req_logit, req_hidden = self.req_classifiers(u_enc_out=u_enc_out, last_hidden=last_hidden,
                                                             z_tm1=req_tm)
                req_logits.append(req_logit)  # batchsize, 1
                req_hiddens.append(req_hidden)  # 1, batchsize, hidden
            req_logits = torch.cat(req_logits, dim=1)  # batchsize, 7
            req_hiddens = torch.cat(req_hiddens, dim=0)  # 7, batchsize, hidden

            res_hiddens = []
            res_logits = []
            belief_hiddens = torch.cat((concat_i_dec_outs, req_hiddens), dim=0)
            for res_idx, res_key in enumerate(requestable_slot):
                res_tm = res_key.unsqueeze(0)
                res_logit, res_hidden = self.res_classifiers(u_enc_out=belief_hiddens,
                                                             last_hidden=req_hiddens[res_idx].unsqueeze(0),
                                                             degree_input=degree_input.unsqueeze(0),
                                                             z_tm1=res_tm)
                res_logits.append(res_logit)
                res_hiddens.append(res_hidden)
            res_logits = torch.cat(res_logits, dim=1)
            res_hiddens = torch.cat(res_hiddens, dim=0)
            requested_7 = kwargs.get('requested_7').squeeze(2).cpu().data.numpy()  # 7, batchsize, 1
            response_7 = kwargs.get('response_7').squeeze(2).cpu().data.numpy()  # 7, batchsize, 1
            pz_dec_outs = torch.cat([pz_dec_outs, req_hiddens, res_hiddens], dim=0)  # seqlen, batchsize, hidden
            z_keyslot_proba = np.concatenate((np.ones((z_input_np.shape[0], batch_size)), requested_7, response_7),
                                             axis=0)  # seqlen, batchsize
            z_keyslot_input_np_ = np.concatenate((z_input_np, requestable_key_np, requestable_slot_np), axis=0)
            pm_dec_proba, m_dec_outs = [], []
            m_length = m_input.size(0)  # Tm
            for t in range(m_length):
                teacher_forcing = toss_(self.teacher_force)
                proba, last_hidden, dec_out = self.m_decoder(pz_dec_outs, u_enc_out, u_input_np, m_tm1,
                                                             degree_input, last_hidden, z_keyslot_input_np_,
                                                             z_keyslot_proba)
                if teacher_forcing:
                    m_tm1 = m_input[t].view(1, -1)
                else:
                    _, m_tm1 = torch.topk(proba, 1)
                    m_tm1 = m_tm1.view(1, -1)
                pm_dec_proba.append(proba)
                m_dec_outs.append(dec_out)
            pm_dec_proba = torch.stack(pm_dec_proba, dim=0)  # [T,B,V]
            return pz_proba, pm_dec_proba, None, req_logits, res_logits
        else:
            i_wordindex = []
            i_dec_outs = []
            for k_index, k_batch in enumerate(k_input):  # batch of one key
                k_tm1 = k_batch.unsqueeze(0)
                k_last_hidden = last_hidden
                if self.num_head == 1:
                    m_idx = -1
                else:
                    m_idx = k_index
                k_dec_outs, k_wordindex, k_last_hidden = \
                    self.bspan_decoder(u_enc_out=u_enc_out, z_tm1=k_tm1, last_hidden=k_last_hidden,
                                       u_input_np=u_input_np,
                                       pv_z_enc_out=pv_z_enc_out,
                                       prev_z_input_np=prev_z_input_np,
                                       u_emb=u_emb, pv_z_emb=pv_z_emb, length=cfg.inf_length,
                                       module_idx=m_idx)
                i_wordindex.append(k_wordindex)
                i_dec_outs.append(torch.cat(k_dec_outs, dim=0))

            pz_dec_outs = torch.cat(i_dec_outs, dim=0)  # seqlen, batchsize, hiddendim
            pz_index = i_wordindex
            pz_index = [(np.asarray(pz_i)).transpose(1, 0) for pz_i in pz_index]  # keysize, batchsize, seqlen

            req_hiddens = []
            req_logits = []
            for req_idx, req_key in enumerate(requestable_key):
                req_tm = req_key.unsqueeze(0)
                if prev_z_input is not None:
                    req_logit, req_hidden = self.req_classifiers(u_enc_out=torch.cat([pv_z_enc_out, u_enc_out], dim=0), last_hidden=last_hidden,
                                                             z_tm1=req_tm)
                else:
                    req_logit, req_hidden = self.req_classifiers(u_enc_out=u_enc_out, last_hidden=last_hidden,
                                                             z_tm1=req_tm)
                req_logits.append(req_logit)  # batchsize, 1
                req_hiddens.append(req_hidden)  # 1, batchsize, hidden
            req_logits = torch.cat(req_logits, dim=1)  # batchsize, 7
            req_hiddens = torch.cat(req_hiddens, dim=0)  # 7, batchsize, hidden

            res_hiddens = []
            res_logits = []
            belief_hiddens = torch.cat((pz_dec_outs, req_hiddens), dim=0)
            for res_idx, res_key in enumerate(requestable_slot):
                res_tm = res_key.unsqueeze(0)
                res_logit, res_hidden = self.res_classifiers(u_enc_out=belief_hiddens,
                                                             last_hidden=req_hiddens[res_idx].unsqueeze(0),
                                                             degree_input=degree_input.unsqueeze(0),
                                                             z_tm1=res_tm)
                res_logits.append(res_logit)
                res_hiddens.append(res_hidden)
            res_logits = torch.cat(res_logits, dim=1)
            res_hiddens = torch.cat(res_hiddens, dim=0)

            pz_dec_outs = torch.cat([pz_dec_outs, req_hiddens, res_hiddens])  # seqlen, batchsize, hidden

            req_out_np = (torch.sigmoid(req_logits)).cpu().data.numpy().transpose(1, 0)  # 7,batchsize
            res_out_np = (torch.sigmoid(res_logits)).cpu().data.numpy().transpose(1, 0)  # 7,batchsize
            z_keyslot_proba = np.concatenate(
                (np.ones((cfg.inf_length * k_input.size(0), batch_size)), req_out_np, res_out_np),
                axis=0)  # seqlen, batchsize

            bspan_index = []  # batchsize, seqlen
            for b_idx in range(k_input.size(1)):
                bspan = []
                for k_idx in range(k_input.size(0)):
                    bspan += [_.cpu().item() for _ in i_wordindex[k_idx][b_idx]]

                for key_idx, proba in zip(requestable_key_np[:, b_idx].tolist(), req_out_np[:, b_idx].tolist()):
                    if proba >= 0.5:
                        bspan.append(key_idx)
                bspan.append(self.vocab.encode('EOS_Z2'))
                bspan_index.append(bspan)

            pz_index = []  # for response decode
            for b_idx in range(k_input.size(1)):
                bspan = []
                for k_idx in range(k_input.size(0)):
                    bspan += i_wordindex[k_idx][b_idx]
                bspan += requestable_key_np[:, b_idx].tolist() + requestable_slot_np[:, b_idx].tolist()
                pz_index.append(bspan)

            if reader != None:
                batch_cons = []
                for b_idx in range(k_input.size(1)):
                    b_con = []
                    for k_idx in range(k_input.size(0)):
                        b = i_wordindex[k_idx][b_idx]
                        eok = constraint_eos[b_idx][k_idx]
                        if eok in b:
                            idx = b.index(eok)
                            cons = b[:idx]
                            cons = [reader.vocab.decode(w) for w in cons]
                            b_con += cons
                    batch_cons.append(b_con)

                db_degrees = []
                if database == None:  # camrest dataset
                    for b in batch_cons:
                        db_result = len(reader.db_search(b))
                        db_degree = [0] * cfg.degree_size
                        db_degree[min(cfg.degree_size - 1, db_result)] = 1.
                        db_degrees.append(db_degree)

                else:  # kvret dataset
                    for b, db in zip(batch_cons, database):
                        b = ' '.join(b)
                        b = b.split(' ; ')
                        db_result = reader.db_degree(b, database)
                        db_degree = [0] * cfg.degree_size
                        db_degree[min(cfg.degree_size - 1, db_result)] = 1.
                        db_degrees.append(db_degree)

                degree_input_np = np.asarray(db_degrees)
                degree_input = cuda_(Variable(torch.from_numpy(degree_input_np).float()))

            if mode == 'test':
                if not self.beam_search:
                    m_output_index = self.greedy_decode(pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden,
                                                        degree_input, pz_index, z_keyslot_proba)

                else:
                    m_output_index = self.beam_search_decode(pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden,
                                                             degree_input, pz_index, z_keyslot_proba)

                return [[_.cpu().item() for _ in l] for l in m_output_index], bspan_index, None

    def bspan_decoder(self, u_enc_out, z_tm1, last_hidden, u_input_np, pv_z_enc_out, prev_z_input_np, u_emb, pv_z_emb,
                      length, module_idx):
        pz_dec_outs = []
        pz_proba = []
        decoded = []
        batch_size = u_enc_out.size(1)
        hiddens = [None] * batch_size
        for t in range(length):
            last_hidden, proba = \
                self.z_decoders[module_idx](u_enc_out=u_enc_out, u_input_np=u_input_np,
                                            z_tm1=z_tm1, last_hidden=last_hidden, pv_z_enc_out=pv_z_enc_out,
                                            prev_z_input_np=prev_z_input_np, u_emb=u_emb, pv_z_emb=pv_z_emb)
            pz_proba.append(proba)
            pz_dec_outs.append(last_hidden)
            z_proba, z_index = torch.topk(proba, 1)  # [B,1]
            z_index = z_index.data.view(-1)
            decoded.append(z_index.clone())
            for i in range(z_index.size(0)):
                if z_index[i] >= cfg.vocab_size:
                    z_index[i] = 2  # unk
            z_np = z_tm1.view(-1).cpu().data.numpy()
            for i in range(batch_size):
                if z_np[i] == self.vocab.encode('EOS_Z2'):
                    hiddens[i] = last_hidden[:, i, :]
            z_tm1 = cuda_(Variable(z_index).view(1, -1))
        for i in range(batch_size):
            if hiddens[i] is None:
                hiddens[i] = last_hidden[:, i, :]
        last_hidden = torch.stack(hiddens, dim=1)
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        decoded = [list(_) for _ in decoded]
        return pz_dec_outs, decoded, last_hidden

    def greedy_decode(self, pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden, degree_input, bspan_index, z_proba):
        decoded = []
        bspan_index_np = pad_sequences(bspan_index).transpose((1, 0))
        for t in range(self.max_ts):
            proba, last_hidden, _ = self.m_decoder(pz_dec_outs, u_enc_out, u_input_np, m_tm1,
                                                   degree_input, last_hidden, bspan_index_np, z_proba)
            mt_proba, mt_index = torch.topk(proba, 1)  # [B,1]
            mt_index = mt_index.data.view(-1)
            decoded.append(mt_index.clone())
            for i in range(mt_index.size(0)):
                if mt_index[i] >= cfg.vocab_size:
                    mt_index[i] = 2  # unk
            m_tm1 = cuda_(Variable(mt_index).view(1, -1))
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        return [list(_) for _ in decoded]

    def beam_search_decode_single(self, pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden, degree_input,
                                  bspan_index, z_proba):
        eos_token_id = self.vocab.encode(cfg.eos_m_token)
        batch_size = pz_dec_outs.size(1)
        if batch_size != 1:
            raise ValueError('"Beam search single" requires batch size to be 1')

        class BeamState:
            def __init__(self, score, last_hidden, decoded, length):
                """
                Beam state in beam decoding
                :param score: sum of log-probabilities
                :param last_hidden: last hidden
                :param decoded: list of *Variable[1*1]* of all decoded words
                :param length: current decoded sentence length
                """
                self.score = score
                self.last_hidden = last_hidden
                self.decoded = decoded
                self.length = length

            def update_clone(self, score_incre, last_hidden, decoded_t):
                decoded = copy.copy(self.decoded)
                decoded.append(decoded_t)
                clone = BeamState(self.score + score_incre, last_hidden, decoded, self.length + 1)
                return clone

        def beam_result_valid(decoded_t, bspan_index):
            decoded_t = [_.view(-1).data[0] for _ in decoded_t]
            req_slots = self.get_req_slots(bspan_index)
            decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_m_token)
            for req in req_slots:
                if req not in decoded_sentence:
                    return False
            return True

        def score_bonus(state, decoded, bspan_index):
            bonus = cfg.beam_len_bonus
            return bonus

        def soft_score_incre(score, turn):
            return score

        finished, failed = [], []
        states = []  # sorted by score decreasingly
        dead_k = 0
        states.append(BeamState(0, last_hidden, [m_tm1], 0))
        bspan_index_np = np.array(bspan_index).reshape(-1, 1)
        for t in range(self.max_ts):
            new_states = []
            k = 0
            while k < len(states) and k < self.beam_size - dead_k:
                state = states[k]
                last_hidden, m_tm1 = state.last_hidden, state.decoded[-1]
                proba, last_hidden, _ = self.m_decoder(pz_dec_outs, u_enc_out, u_input_np, m_tm1, degree_input,
                                                       last_hidden, bspan_index_np, z_proba)

                proba = torch.log(proba)
                mt_proba, mt_index = torch.topk(proba, self.beam_size - dead_k)  # [1,K]
                for new_k in range(self.beam_size - dead_k):
                    score_incre = soft_score_incre(mt_proba[0][new_k].data[0], t) + score_bonus(state,
                                                                                                mt_index[0][new_k].data[
                                                                                                    0], bspan_index)
                    if len(new_states) >= self.beam_size - dead_k and state.score + score_incre < new_states[-1].score:
                        break
                    decoded_t = mt_index[0][new_k]
                    if decoded_t.data[0] >= cfg.vocab_size:
                        decoded_t.data[0] = 2  # unk
                    if self.vocab.decode(decoded_t.data[0]) == cfg.eos_m_token:
                        if beam_result_valid(state.decoded, bspan_index):
                            finished.append(state)
                            dead_k += 1
                        else:
                            failed.append(state)
                    else:
                        decoded_t = decoded_t.view(1, -1)
                        new_state = state.update_clone(score_incre, last_hidden, decoded_t)
                        new_states.append(new_state)

                k += 1
            if self.beam_size - dead_k < 0:
                break
            new_states = new_states[:self.beam_size - dead_k]
            new_states.sort(key=lambda x: -x.score)
            states = new_states

            if t == self.max_ts - 1 and not finished:
                finished = failed
                print('FAIL')
                if not finished:
                    finished.append(states[0])

        finished.sort(key=lambda x: -x.score)
        decoded_t = finished[0].decoded
        decoded_t = [_.view(-1).data[0] for _ in decoded_t]
        decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_m_token)
        print(decoded_sentence)
        generated = torch.cat(finished[0].decoded, dim=1).data  # [B=1, T]
        return generated

    def beam_search_decode(self, pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden, degree_input, bspan_index, z_proba):
        vars = torch.split(pz_dec_outs, 1, dim=1), torch.split(u_enc_out, 1, dim=1), torch.split(
            m_tm1, 1, dim=1), torch.split(last_hidden, 1, dim=1), torch.split(degree_input, 1, dim=0)
        decoded = []
        for i, (pz_dec_out_s, u_enc_out_s, m_tm1_s, last_hidden_s, degree_input_s) in enumerate(zip(*vars)):
            decoded_s = self.beam_search_decode_single(pz_dec_out_s, u_enc_out_s, m_tm1_s,
                                                       u_input_np[:, i].reshape((-1, 1)),
                                                       last_hidden_s, degree_input_s, bspan_index[i],
                                                       z_proba[:, i].reshape((-1, 1)))
            decoded.append(decoded_s)
        return [list(_.view(-1)) for _ in decoded]

    def supervised_loss(self, pz_proba, pm_dec_proba, z_input, m_input, requested_logit, response_logit,
                        requested_input, response_input, loss_weights=[1., 1., 1., 1.]):
        pz_proba, pm_dec_proba = pz_proba[:, :, :cfg.vocab_size].contiguous(), pm_dec_proba[:, :,
                                                                               :cfg.vocab_size].contiguous()
        pr_loss = self.pr_loss(pz_proba.view(-1, pz_proba.size(2)), z_input.view(-1))
        m_loss = self.dec_loss(pm_dec_proba.view(-1, pm_dec_proba.size(2)), m_input.view(-1))
        requested_loss = self.multilabel_loss(requested_logit, requested_input)
        response_loss = self.multilabel_loss(response_logit, response_input)

        loss = loss_weights[0] * pr_loss + loss_weights[1] * requested_loss + loss_weights[2] * response_loss + \
               loss_weights[3] * m_loss
        return loss, pr_loss, m_loss, requested_loss, response_loss

    def self_adjust(self, epoch):
        pass
