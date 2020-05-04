# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from transformers import BertModel
from transformers.modeling_bert import BertPreTrainedModel


class BertForDistantRE(BertPreTrainedModel):
    
    def __init__(self, config, num_labels, dropout=0.1, bag_attn=False):
        super(BertForDistantRE, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.We = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()
        self.classifier = nn.Linear(3*config.hidden_size, num_labels)
        self.bag_attn = bag_attn
        if bag_attn:
            self.Wo = nn.Linear(3*config.hidden_size, 3*config.hidden_size)
        self.softmax = nn.Softmax(-1)
        self.init_weights()
    
    def bag_attention_logits(self, bag, labels=None, is_train=True):
        if is_train:
            query = labels.unsqueeze(1) # B x 1
            attn_M = self.classifier.weight.data[query] # B x 1 x H
            attn_s = (bag * attn_M).sum(-1) # (B x G x H) * (B x 1 x H) -> B x G x H -> B x G
            attn_s = self.softmax(attn_s) # B x G
            bag = (attn_s.unsqueeze(-1) * bag).sum(1) # (B x G x 1) * (B x G x H) -> B x G x H -> B x H
            logits = self.classifier(self.dropout(self.act(bag))) # B x C
        else:
            attn_M = bag.matmul(self.classifier.weight.data.transpose(0, 1)) # (B x G x H) * (H x C) -> B x G x C
            attn_s = self.softmax(attn_M.transpose(-1, -2)) # B x C x G
            bag = attn_s.bmm(bag) # (B x C x G) * (B x G x H) -> B x C x H
            logits = self.classifier(self.dropout(self.act(bag))) # B x C x C
            logits = logits.diagonal(dim1=1, dim2=2) # B x C
        return logits
    
    def forward(self,
                input_ids,
                entity_ids=None,
                attention_mask=None,
                labels=None,
                is_train=True):
        ## PART-I: Encode the sequence with BERT
        B, G, L = input_ids.shape
        
        input_ids = input_ids.view(B*G, -1)
        attention_mask = attention_mask = attention_mask.view(B*G, -1)
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = outputs[0], outputs[1]
        
        sequence_output = sequence_output.view(B, G, L, -1).clone() # B x G x L x H
        pooled_output = pooled_output.view(B, G, -1).clone() # B x G x H
        
        ## PART-II: Get e1 and e2 hidden representations
        e1_mask = (entity_ids == 1).float() # locations of e1 entity
        e1 = sequence_output * e1_mask.unsqueeze(-1) # B x G x L x H
        e1 = e1.sum(2) / e1_mask.sum(2).unsqueeze(-1) # Empty sequences will have all zeros
        e1 = self.We(self.dropout(self.act(e1))) # B x G x H
        
        # Similarly for e2 entity
        e2_mask = (entity_ids == 2).float()
        e2 = sequence_output * e2_mask.unsqueeze(-1)
        e2 = e2.sum(2) / e2_mask.sum(2).unsqueeze(-1)
        e2 = self.We(self.dropout(self.act(e2))) # B x G x H
        
        # PART-III: Average bag aggregation and relation classifier
        r_h = torch.cat((pooled_output, e1, e2), -1) # B x G x 3H
        if self.bag_attn:
            r_h = self.Wo(self.dropout(self.act(r_h)))
            logits = self.bag_attention_logits(r_h, labels, is_train)
        else:
            r_h = r_h.sum(1) / G
            logits = self.classifier(self.dropout(self.act(r_h)))
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels)
            outputs = (loss,) + outputs
        
        return outputs  # (loss), scores, (hidden_states), (attentions)
