import numpy as np
from random import sample
import torch.nn.init as init
from torch.autograd import Variable
import time
from numbers import Number
import math
import logging
from typing import Optional, Tuple
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch.nn import L1Loss, MSELoss
from torch.autograd import Function
from math import pi, log
from functools import wraps
import os
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.activations import gelu, gelu_new
from transformers import BertConfig
import numpy as np
import torch.optim as optim
from itertools import chain
from model_utils import *
from utils import *
import torch
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from math import pi, log
from functools import wraps
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from modules.transformer import TransformerEncoder


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
]


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
}
   
class BERT_TM(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.config.output_hidden_states=True
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # fused_embedding = embedding_output

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_sequence_output = encoder_outputs[0]# 36*60*768:bsz*msl*dim
        # print('last_sequence_output_shape', last_sequence_output.shape)# [256,50,768]
        return last_sequence_output




class HME(BertPreTrainedModel):
    def __init__(self, config, args = None):
        super().__init__(config)
        self.bert = BERT_TM(config, args)
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.TEXT_DIM, args.ACOUSTIC_DIM, args.VISUAL_DIM
        self.d_l = self.d_a = self.d_v = args.d_l
        self.args = args
        self.bsz = args.train_batch_size 
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.latent_depth = args.latent_layers
        self.activation = nn.ReLU()
        self.missing_rate = args.missing_rate#
        self.similarity_threshold = args.similarity_threshold
        self.num_latents = args.num_latents
        self.hyper_depth = args.hyper_depth
        self.output_dim = args.output_dim
        self.LB = args.uncertainty_LB
        self.UB = args.uncertainty_UB

        # within-modality interactions
        self.l2l = self.Get_Modality_Network(modality='ll')
        self.a2a = self.Get_Modality_Network(modality='aa')
        self.v2v = self.Get_Modality_Network(modality='vv')
        
        # cross-modality interactions
        self.l2a = self.get_hyper_modality(self_type='l2a')
        self.l2v = self.get_hyper_modality(self_type='l2v')
        self.a2l = self.get_hyper_modality(self_type='a2l')
        self.a2v = self.get_hyper_modality(self_type='a2v')
        self.v2l = self.get_hyper_modality(self_type='v2l')
        self.v2a = self.get_hyper_modality(self_type='v2a')

        self.hyper_l = self.get_hyper_modality(self_type='hyper_l')
        self.hyper_a = self.get_hyper_modality(self_type='hyper_a')
        self.hyper_v = self.get_hyper_modality(self_type='hyper_v')

        self.latent_sample = self.get_latent_network(self_type='sam')
        self.latent_text = self.get_latent_network(self_type='mod_l')
        self.latent_audio = self.get_latent_network(self_type='mod_a')
        self.latent_video = self.get_latent_network(self_type='mod_v')

        # 1. Temporal convolutional layers
        self.proj_l_1 = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)#
        self.proj_a_1 = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)#
        self.proj_v_1 = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)#

        # VIB estimation
        self.VIB_estimate_l_p = VIB_estimate(self.d_l, self.d_l, self.output_dim)
        self.VIB_estimate_a_p = VIB_estimate(self.d_a, self.d_a, self.output_dim)
        self.VIB_estimate_v_p = VIB_estimate(self.d_v, self.d_v, self.output_dim)
        
        self.pool_audio = nn.AdaptiveMaxPool1d(1)
        self.pool_text = nn.AdaptiveMaxPool1d(1)
        self.pool_video = nn.AdaptiveMaxPool1d(1)
        self.dropout = args.dropout
        
        # Integrate the hyper into modality
        self.ATTN_T = Attention_Uncertainty_2(self.d_l, self.d_l, self.LB, self.UB)
        self.ATTN_A = Attention_Uncertainty_2(self.d_a, self.d_a, self.LB, self.UB)
        self.ATTN_V = Attention_Uncertainty_2(self.d_v, self.d_v, self.LB, self.UB)

        self.ATTN_hyper = Attention_Uncertainty_3(self.d_l, self.d_l)
        
        encoder_layer_all = nn.TransformerEncoderLayer(d_model=self.d_l, nhead=self.num_heads)
        self.transformer_encoder_all = nn.TransformerEncoder(encoder_layer_all, num_layers=2)

        self.fusion_all = nn.Sequential()
        self.fusion_all.add_module('fusion_layer_all', nn.Linear(in_features=self.d_l*4, out_features=self.d_l*2))
        self.fusion_all.add_module('fusion_layer_all_dropout', nn.Dropout(self.dropout))
        self.fusion_all.add_module('fusion_layer_all_activation', self.activation)
        self.fusion_all.add_module('fusion_layer_all_all', nn.Linear(in_features=self.d_l*2, out_features=self.output_dim))  
        self.init_weights()
        
    def Get_Modality_Network(self, modality='l'):
        if modality in ['ll', 'al', 'vl']:
            attn_dropout = self.attn_dropout
        elif modality in ['aa', 'la', 'va']:
            attn_dropout = self.attn_dropout_a
        elif modality in ['vv', 'lv', 'av']:
            attn_dropout = self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")
            
        return TransformerEncoder(embed_dim=self.d_l,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=False)

    def get_latent_network(self, self_type='sam',layers=-1):
        assert self_type in ['sam', 'mod_l', 'mod_a', 'mod_v']
        
        return LatentTransformer(num_latents=self.num_latents,
                                 latent_dim=self.d_l,
                                 input_dim=self.d_l,
                                 depth=self.latent_depth,
                                 heads=self.num_heads,
                                 latent_heads=self.num_heads,
                                 latent_dim_head=self.d_l//self.num_heads,
                                 dim_head=self.d_l//self.num_heads,
                                 )

    def get_hyper_modality(self, self_type='sam',layers=-1):
        assert self_type in ['l2a', 'l2v', 'a2l', 'a2v', 'v2l', 'v2a', 'hyper_l', 'hyper_a', 'hyper_v']
        # input_dim, depth, heads, dim_head, latent_heads, latent_dim_head
        return Encoder_Layer(input_dim=self.d_l,
                             depth=self.hyper_depth,
                             heads=self.num_heads,
                             dim_head=self.d_l//self.num_heads,
                             latent_heads=self.num_heads,
                             latent_dim_head=self.d_l//self.num_heads,
        )

    def get_similar_embeddings(self, embeddings, alpha=0.8):
        batch_size, emb_dim = embeddings.size()
        device = embeddings.device
        # Compute pairwise cosine similarity
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        # Create masks to avoid self-similarity
        mask = torch.eye(batch_size, device=device).bool()
        cosine_sim.masked_fill_(mask, float('-inf'))
        # Find indices where similarity is greater than or equal to alpha
        similar_mask = cosine_sim >= alpha
        # Initialize mean_embeddings tensor
        mean_embeddings = torch.zeros(batch_size, emb_dim, device=device)
        # Compute mean embeddings for each group
        for i in range(batch_size):
            similar_indices = similar_mask[i].nonzero(as_tuple=True)[0]
            if len(similar_indices) > 0:
                similar_embeddings = embeddings[similar_indices]
                mean_embeddings[i] = similar_embeddings.mean(dim=0)
            else:
                # If no embeddings have similarity greater than or equal to alpha, use zero vector
                mean_embeddings[i] = torch.zeros(emb_dim, device=device)
        
        # Handle zero vectors and non-zero vectors with similarity less than alpha
        non_zero_embeddings = embeddings[embeddings.sum(dim=1) != 0]
        if non_zero_embeddings.size(0) > 0:
            mean_non_zero_embeddings = non_zero_embeddings.mean(dim=0, keepdim=True)
        else:
            mean_non_zero_embeddings = torch.zeros(1, emb_dim, device=device)
        
        # Use mean_non_zero_embeddings for zero vectors and non-zero vectors with similarity less than alpha
        mean_embeddings = torch.where(
            (embeddings.sum(dim=1, keepdim=True) == 0) | (~similar_mask.any(dim=1, keepdim=True)),
            mean_non_zero_embeddings,
            mean_embeddings
        )
        
        return mean_embeddings 

    
    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        ps,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):
        
        texts = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        x_l, x_a, x_v = generate_missing_modalities(ps, texts, acoustic, visual)
        x_l = x_l.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        
        # Note that the input is incomplete
        proj_x_l = self.proj_l_1(x_l)
        proj_x_a = self.proj_a_1(x_a)
        proj_x_v = self.proj_v_1(x_v)
        
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        # interactions between ori_representations
        trans_text = self.l2l(proj_x_l,proj_x_l,proj_x_l)
        trans_text = trans_text.permute(1,2,0)
        trans_audio = self.a2a(proj_x_a,proj_x_a,proj_x_a)
        trans_audio = trans_audio.permute(1,2,0)
        trans_video = self.v2v(proj_x_v,proj_x_v,proj_x_v)
        trans_video = trans_video.permute(1,2,0)
        
        last_x_l = self.pool_text(trans_text).squeeze()
        last_x_a = self.pool_audio(trans_audio).squeeze()
        last_x_v = self.pool_video(trans_video).squeeze()

        # enhanced representations
        similar_texts = self.get_similar_embeddings(last_x_l, self.similarity_threshold)
        similar_audios = self.get_similar_embeddings(last_x_a, self.similarity_threshold)
        similar_videos = self.get_similar_embeddings(last_x_v, self.similarity_threshold)

        modality_all = torch.stack([last_x_l, last_x_a, last_x_v], dim=1)
        
        eh_l = self.latent_text(similar_texts)# 
        eh_a = self.latent_audio(similar_audios)# 
        eh_v = self.latent_video(similar_videos)# 
        enhance_samples = self.latent_sample(modality_all)
        
        # FUSION OF HYPERS
        hyper_l2a = self.l2a(eh_l.unsqueeze(dim=1), eh_a.unsqueeze(dim=1))
        hyper_l2v = self.l2v(eh_l.unsqueeze(dim=1), eh_v.unsqueeze(dim=1))
        hyper_a2l = self.a2l(eh_a.unsqueeze(dim=1), eh_l.unsqueeze(dim=1))
        hyper_a2v = self.a2v(eh_a.unsqueeze(dim=1), eh_v.unsqueeze(dim=1))
        hyper_v2l = self.v2l(eh_v.unsqueeze(dim=1), eh_l.unsqueeze(dim=1))
        hyper_v2a = self.v2a(eh_v.unsqueeze(dim=1), eh_a.unsqueeze(dim=1))

        # Fusions 
        hyper_ls = self.hyper_l(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_l2a, hyper_l2v), dim=1))
        hyper_as = self.hyper_a(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_a2l, hyper_a2v), dim=1))
        hyper_vs = self.hyper_v(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_v2l, hyper_v2a), dim=1))
        
        # Use the VIB to maintain the essential information
        (mu_l_p, std_l_p), encoding_l_p, logits_l_p, info_loss_l_p = self.VIB_estimate_l_p(hyper_ls)
        (mu_a_p, std_a_p), encoding_a_p, logits_a_p, info_loss_a_p = self.VIB_estimate_a_p(hyper_as)
        (mu_v_p, std_v_p), encoding_v_p, logits_v_p, info_loss_v_p = self.VIB_estimate_v_p(hyper_vs)
        
        # Fuse the hyper-modalities
        hyper_all = self.ATTN_hyper(encoding_l_p, encoding_a_p, encoding_v_p, std_l_p, std_a_p, std_v_p)
        
        # Fuse the hyper-modalities into ori_representations
        refine_ls = self.ATTN_T(last_x_l, encoding_l_p, std_l_p, self.LB, self.UB)
        refine_as = self.ATTN_A(last_x_a, encoding_a_p, std_a_p, self.LB, self.UB)
        refine_vs = self.ATTN_V(last_x_v, encoding_v_p, std_v_p, self.LB, self.UB)

        all_embed = torch.stack((refine_ls, refine_as, refine_vs, hyper_all), dim=0)
        hidden = self.transformer_encoder_all(all_embed)
        hidden = torch.cat((hidden[0], hidden[1], hidden[2], hidden[3]), dim=1)
        output = self.fusion_all(hidden)
        info_losses = (info_loss_l_p + info_loss_a_p + info_loss_v_p) / 3.0
        res = {
                "logits_l_p": logits_l_p,
                "logits_a_p": logits_a_p,
                "logits_v_p": logits_v_p,
                "predictions": output,
                "info_losses": info_losses,
        }

        
        return res
        
    def test(
        self,
        input_ids,
        visual,
        acoustic,
        ps,
        label_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):

        
        texts = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        x_l, x_a, x_v = generate_missing_modalities(ps, texts, acoustic, visual)
        x_l = x_l.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        
        proj_x_l = self.proj_l_1(x_l)
        proj_x_a = self.proj_a_1(x_a)
        proj_x_v = self.proj_v_1(x_v)
        
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        # interactions between ori_representations
        trans_text = self.l2l(proj_x_l,proj_x_l,proj_x_l)
        trans_text = trans_text.permute(1,2,0)
        trans_audio = self.a2a(proj_x_a,proj_x_a,proj_x_a)
        trans_audio = trans_audio.permute(1,2,0)
        trans_video = self.v2v(proj_x_v,proj_x_v,proj_x_v)
        trans_video = trans_video.permute(1,2,0)
        
        last_x_l = self.pool_text(trans_text).squeeze()# [24,30,1]
        last_x_a = self.pool_audio(trans_audio).squeeze()# [24,30,1]
        last_x_v = self.pool_video(trans_video).squeeze()# [24,30,1]

        # enhanced representations
        similar_texts = self.get_similar_embeddings(last_x_l, self.similarity_threshold)
        similar_audios = self.get_similar_embeddings(last_x_a, self.similarity_threshold)
        similar_videos = self.get_similar_embeddings(last_x_v, self.similarity_threshold)

        modality_all = torch.stack([last_x_l, last_x_a, last_x_v], dim=1)# [BSZ,DIM,DIM]
        
        eh_l = self.latent_text(similar_texts)# 
        eh_a = self.latent_audio(similar_audios)# 
        eh_v = self.latent_video(similar_videos)# 
        enhance_samples = self.latent_sample(modality_all)
        
        # FUSION OF HYPERS
        hyper_l2a = self.l2a(eh_l.unsqueeze(dim=1), eh_a.unsqueeze(dim=1))
        hyper_l2v = self.l2v(eh_l.unsqueeze(dim=1), eh_v.unsqueeze(dim=1))
        hyper_a2l = self.a2l(eh_a.unsqueeze(dim=1), eh_l.unsqueeze(dim=1))
        hyper_a2v = self.a2v(eh_a.unsqueeze(dim=1), eh_v.unsqueeze(dim=1))
        hyper_v2l = self.v2l(eh_v.unsqueeze(dim=1), eh_l.unsqueeze(dim=1))
        hyper_v2a = self.v2a(eh_v.unsqueeze(dim=1), eh_a.unsqueeze(dim=1))

        # Fusions 
        hyper_ls = self.hyper_l(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_l2a, hyper_l2v), dim=1))
        hyper_as = self.hyper_a(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_a2l, hyper_a2v), dim=1))
        hyper_vs = self.hyper_v(enhance_samples.unsqueeze(dim=1), torch.stack((hyper_v2l, hyper_v2a), dim=1))
        
        # Use the VIB to maintain the essential information
        (mu_l_p, std_l_p), encoding_l_p, logits_l_p, info_loss_l_p = self.VIB_estimate_l_p(hyper_ls)
        (mu_a_p, std_a_p), encoding_a_p, logits_a_p, info_loss_a_p = self.VIB_estimate_a_p(hyper_as)
        (mu_v_p, std_v_p), encoding_v_p, logits_v_p, info_loss_v_p = self.VIB_estimate_v_p(hyper_vs)
        
        # Fuse the hyper-modalities
        hyper_all = self.ATTN_hyper(encoding_l_p, encoding_a_p, encoding_v_p, std_l_p, std_a_p, std_v_p)
        
        # Fuse the hyper-modalities into ori_representations
        refine_ls = self.ATTN_T(last_x_l, encoding_l_p, std_l_p, self.LB, self.UB)
        refine_as = self.ATTN_A(last_x_a, encoding_a_p, std_a_p, self.LB, self.UB)
        refine_vs = self.ATTN_V(last_x_v, encoding_v_p, std_v_p, self.LB, self.UB)

        all_embed = torch.stack((refine_ls, refine_as, refine_vs, hyper_all), dim=0)
        hidden = self.transformer_encoder_all(all_embed)
        hidden = torch.cat((hidden[0], hidden[1], hidden[2], hidden[3]), dim=1)
        output = self.fusion_all(hidden)
        info_losses = (info_loss_l_p + info_loss_a_p + info_loss_v_p) / 3.0
        res = {
                "logits_l_p": logits_l_p,
                "logits_a_p": logits_a_p,
                "logits_v_p": logits_v_p,
                "predictions": output,
                "info_losses": info_losses,
        }

        return res
       