# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. """

from collections import defaultdict
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from modules.activations import ACT2FN
from modules.modeling_outputs import BaseModelOutput

from transformers import PreTrainedModel, logging, BertConfig, BertPreTrainedModel, BertModel


class MBertForAlign(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        src_input_ids=None,
        src_attention_mask=None,
        tgt_input_ids=None,
        tgt_attention_mask=None,
        word_bpe_align=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head_mask=None,
        simalign_method=False,
        self_train=False,
        co_train=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        src_encoder_out = self.bert(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask.float(),
            head_mask=head_mask,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        tgt_encoder_out = self.bert(
            input_ids=tgt_input_ids,
            attention_mask=tgt_attention_mask.float(),
            head_mask=head_mask,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        src_align_out = src_encoder_out[0]
        tgt_align_out = tgt_encoder_out[0]

        # 将encoder out 输入到对齐头中
        atten_mask_src = (1 - ((src_input_ids != 101) & (src_input_ids != 102) & src_attention_mask)[:, None, :].float()) * -10000
        atten_mask_tgt = (1 - ((tgt_input_ids != 101) & (tgt_input_ids != 102) & tgt_attention_mask)[:, None, :].float()) * -10000
        
        if self_train:
            with torch.no_grad():
                # 采用自训练的方式
                if not simalign_method:
                    # dot product + softmax
                    bpe_sim_guides = torch.bmm(src_align_out, tgt_align_out.transpose(1,2))

                    attention_scores_src = bpe_sim_guides + atten_mask_tgt
                    attention_scores_tgt = bpe_sim_guides + atten_mask_src.transpose(-1, -2)
                    
                    attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src)
                    attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt)
                    threshold = 0.001
                    align_matrix = (attention_probs_src > threshold) * (attention_probs_tgt > threshold)
                    
                    guides = torch.nonzero(align_matrix)
                    word_bpe_align = torch.cat([guides[:, 0][:, None] * align_matrix.size(1) + guides[:, 1][:, None], guides[:, 2][:, None]], dim=-1)
                else:
                    # cosine + max
                    # 将encoder out 输入到对齐头中
                    atten_mask_src_guides = ((src_input_ids != 101) & (src_input_ids != 102) & src_attention_mask).float()
                    atten_mask_tgt_guides = ((tgt_input_ids != 101) & (tgt_input_ids != 102) & tgt_attention_mask).float()

                    # mask </s> and langcodes
                    cosine_sim = self.get_cosine_sim(src_align_out * atten_mask_src_guides.unsqueeze(-1),
                                                     tgt_align_out * atten_mask_tgt_guides.unsqueeze(-1), False)
                    # distortion trick
                    distortion_cosine_sim = self.apply_distortion(cosine_sim, distortion=0.0)
                    # get_align_matrix
                    forward_align, backward_align = self.get_align_matrix(distortion_cosine_sim)
                    align_matrix = forward_align * backward_align
                    guides = torch.nonzero(align_matrix)
                    word_bpe_align = torch.cat([guides[:, 0][:, None] * align_matrix.size(1) + guides[:, 1][:, None], guides[:, 2][:, None]], dim=-1)

        if not simalign_method:
            # dot product for awesome method
            bpe_sim = torch.bmm(src_align_out, tgt_align_out.transpose(1,2))
        else:
            # cosine similarity for simalign
            bpe_sim = self.get_cosine_sim(src_align_out, tgt_align_out)
        
        # 还是应该softmax输出
        attention_scores_src = bpe_sim + atten_mask_tgt     # bs * slen * tlen
        attention_scores_tgt = bpe_sim + atten_mask_src.transpose(-1, -2)

        attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src)
        attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt)
        
        len_src = (atten_mask_src==0).sum(dim=-1)
        len_tgt = (atten_mask_tgt==0).sum(dim=-1)
        loss_src = attention_probs_src / len_src.unsqueeze(-1) # token num
        loss_tgt = attention_probs_tgt / len_tgt.unsqueeze(-1)

        loss_src = loss_src.view(-1, loss_src.size(-1))[word_bpe_align[:, 0], word_bpe_align[:, 1]].sum() / len_src.size(0) # sentence num
        loss_tgt = loss_tgt.view(-1, loss_tgt.size(-1))[word_bpe_align[:, 0], word_bpe_align[:, 1]].sum() / len_tgt.size(0)

        # loss = nn.functional.cross_entropy(bpe_sim.view(-1, bpe_sim.size(-1))[word_bpe_align[:, 0]], word_bpe_align[:, 1])
        loss = -loss_src -loss_tgt

        co_loss = 0
        if co_train:
            min_len = torch.min(len_src, len_tgt)
            trace = torch.matmul(attention_probs_src, (attention_probs_tgt).transpose(-1, -2)).squeeze(1)
            trace = torch.einsum('bii->b', trace)
            co_loss = -torch.mean(trace/min_len)
        return loss+co_loss
    def get_para_align(
        self,
        src_input_ids=None,
        src_attention_mask=None,
        src_b2w_map=None,
        tgt_input_ids=None,
        tgt_attention_mask=None,
        tgt_b2w_map=None,
        gold=None,
        threshold=0.001, 
        bpe_level=False,
    ):
        with torch.no_grad():
            src_encoder_out = self.bert(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask.float(),
                head_mask=None,
                inputs_embeds=None,
                output_hidden_states=True,
            )
            tgt_encoder_out = self.bert(
                input_ids=tgt_input_ids,
                attention_mask=tgt_attention_mask.float(),
                head_mask=None,
                inputs_embeds=None,
                output_hidden_states=True,
            )
            
            # 将encoder out 输入到对齐头中
            atten_mask_src = (1 - ((src_input_ids != 101) & (src_input_ids != 102) & src_attention_mask)[:, None, None, :].float()) * -10000
            atten_mask_tgt = (1 - ((tgt_input_ids != 101) & (tgt_input_ids != 102) & tgt_attention_mask)[:, None, None, :].float()) * -10000

            # 残差链接静态embedding
            src_align_out = src_encoder_out[0]
            tgt_align_out = tgt_encoder_out[0]

            bpe_sim = torch.bmm(src_align_out, tgt_align_out.transpose(1,2))

        # 根据余弦相似度将抽取单词级别的对齐
        
        attention_scores_src = bpe_sim.unsqueeze(1) + atten_mask_tgt
        attention_scores_tgt = bpe_sim.unsqueeze(1) + atten_mask_src.transpose(-1, -2)
        
        attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src)
        attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt)

        mask_src_tgt = (((src_input_ids != 101) & (src_input_ids != 102) & src_attention_mask)[:,:,None])&(((tgt_input_ids != 101) & (tgt_input_ids != 102) & tgt_attention_mask)[:,None,:])

        align_matrix = (attention_probs_src > threshold) * (attention_probs_tgt > threshold)
        align_matrix = align_matrix.squeeze(1)

        # word_aligns = []
        # softmax_threshold = threshold
        len_src = (atten_mask_src==0).sum(dim=-1).unsqueeze(-1)
        len_tgt = (atten_mask_tgt==0).sum(dim=-1).unsqueeze(-1)
        
       
        attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src / torch.sqrt(len_src.float()))
        attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt / torch.sqrt(len_tgt.float()))

        # bpe_sim = (2 * attention_probs_src * attention_probs_tgt) / (attention_probs_src + attention_probs_tgt + 1e-9)
        # bpe_sim = bpe_sim.squeeze(1)
        # align_matrix = bpe_sim>threshold
        
        # # 将bpe级别的对齐转到单词级别的对齐

        word_aligns = []
        softmax_threshold = threshold
        for idx, (line_align, b2w_src, b2w_tgt) in enumerate(zip(align_matrix, src_b2w_map, tgt_b2w_map)):
            
            aligns = dict()
            non_specials = torch.where(line_align)

            for i, j in zip(*non_specials):
                # 真实词对
                if not bpe_level:
                    word_pair = (src_b2w_map[idx][i-1].item(), tgt_b2w_map[idx][j-1].item())

                    if not word_pair in aligns:
                        aligns[word_pair] = bpe_sim[idx][i, j].item()
                    else:
                        aligns[word_pair] = max(aligns[word_pair], bpe_sim[idx][i, j].item())
                else:

                    aligns[(i.item()-1, j.item()-1)] = bpe_sim[idx][i, j].item()
            word_aligns.append(aligns)
        return word_aligns

    def simalign(
        self,
        src_input_ids=None,
        src_attention_mask=None,
        src_b2w_map=None,
        tgt_input_ids=None,
        tgt_attention_mask=None,
        tgt_b2w_map=None,
        gold=None,
        threshold=1e-10, 
        bpe_level=False, 
        is_eval=False,
        mono_test=False,
        ):
        
        with torch.no_grad():
            src_encoder_out = self.bert(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask.float(),
                head_mask=None,
                inputs_embeds=None,
                output_hidden_states=True,
            )
            tgt_encoder_out = self.bert(
                input_ids=tgt_input_ids,
                attention_mask=tgt_attention_mask.float(),
                head_mask=None,
                inputs_embeds=None,
                output_hidden_states=True,
            )

            # 将encoder out 输入到对齐头中
            atten_mask_src = ((src_input_ids != 101) & (src_input_ids != 102) & src_attention_mask).float()
            atten_mask_tgt = ((tgt_input_ids != 101) & (tgt_input_ids != 102) & tgt_attention_mask).float()


            src_align_out = src_encoder_out[0]
            tgt_align_out = tgt_encoder_out[0]

            # mask </s> and langcodes
            src_align_out = src_align_out * atten_mask_src.unsqueeze(-1)
            tgt_align_out = tgt_align_out * atten_mask_tgt.unsqueeze(-1)
            cosine_sim = self.get_cosine_sim(src_align_out, tgt_align_out, not is_eval)
            # distortion trick
            distortion_cosine_sim = self.apply_distortion(cosine_sim, distortion=0.0)
            # get_align_matrix
            forward_align, backward_align = self.get_align_matrix(distortion_cosine_sim)
        if not mono_test:
            inter_align = forward_align * backward_align
        else:
            inter_align = forward_align
        outs = []
        # 转化为 word 级别的对齐再进行输出
        for i, line_out in enumerate(inter_align):
            out = set()
            
            for j, pair in enumerate(torch.nonzero(line_out).tolist()):
                if line_out[pair[0], pair[1]] > threshold:

                    out.add((src_b2w_map[i][pair[0] - 1].item(), tgt_b2w_map[i][pair[1 ]- 1].item()))
            outs.append(sorted(out))

        return outs
        
    def get_cosine_sim(
        self,
        tensor_1, 
        tensor_2,
        expand=True,
    ):
        # 计算两个向量的余弦相似度

        # 向量分别做归一化
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        cosine_sim = torch.bmm(normalized_tensor_1, normalized_tensor_2.transpose(1,2))

        # 将nan部分的余弦相似度替换为-1
        cosine_sim = (cosine_sim.masked_fill(torch.isnan(cosine_sim), -1) + 1) / 2
        #cosine_sim= cosine_sim.masked_fill(torch.isnan(cosine_sim), -1)
        if expand:
            #expand_lambda = tensor_1.norm(dim=-1, keepdim=True).mean() * tensor_2.norm(dim=-1, keepdim=True).mean()
            expand_lambda = 595.0
        else: 
            expand_lambda = 1.0
        return cosine_sim * expand_lambda

    def apply_distortion(
        self, 
        sim,
        distortion=0.0
    ):
        shape = sim.shape
        if (shape[1] < 2 or shape[2]) < 2 or distortion <= 0.0:
            return sim
        # 每一句的distortion都是不同的
        src_lens = (sim!=0).sum(dim=-2)[:, 0]
        tgt_lens = (sim!=0).sum(dim=-1)[:, 0]
        dists = []

        for i, j in zip(src_lens, tgt_lens):
            s_x, s_y = i.item(), j.item()

            dist = torch.zeros_like(sim[0])
            pos_x = torch.Tensor([[y / float(s_y - 1) for y in range(s_y)] for x in range(s_x)]).type_as(sim)
            pos_y = torch.Tensor([[y / float(s_x - 1) for y in range(s_x)] for x in range(s_y)]).type_as(sim)

            dist[1:s_x+1, 1:s_y+1] = 1.0 - ((pos_x - pos_y.transpose(0, 1)) ** 2) * distortion
            dists.append(dist.unsqueeze(0))
        dist_mask = torch.cat(dists, dim=0)
        return dist_mask * sim
    def get_align_matrix(
        self,
        sim,
    ):

        bs, m, n = sim.shape
        zero_mask = (sim != 0)
        forward_align = torch.eye(n)[sim.max(-1)[1].view(-1)].view(bs, m, n).to(sim.device)
        backward_align = torch.eye(m)[sim.max(-2)[1].view(-1)].view(bs, n, m).to(sim.device)

        return forward_align * zero_mask, backward_align.transpose(1, 2) * zero_mask

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

