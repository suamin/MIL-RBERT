# -*- coding: utf-8 -*-

import torch

# ------------------------------------------------------
# Work arounds to errors mentioned here:
# https://github.com/pytorch/pytorch/issues/973

torch.multiprocessing.set_sharing_strategy('file_system')

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# FIXME: These fixes are not stable and may or may not
# work. The num of worker and chunksize influence it
# as well (somehow!). One can just remove it altogether
# and create features serially (much slower but stable).

# -------------------------------------------------------

import logging
import config
import functools
import copy

from transformers import BertTokenizer
from concurrent.futures import ProcessPoolExecutor
from data_utils import JsonlReader, read_entities, read_relations

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_jsonl(jsonl, tokenizer, entity2idx, relation2idx, max_seq_length=128, 
                   e1_tok="$", e2_tok="^", entity_start=False):
    # Group
    src, tgt = jsonl["group"]
    relation = jsonl["relation"]
    input_ids = list()
    entity_ids = list()
    attention_mask = list()
    
    for sent in jsonl["sentences"]:
        encoded = tokenizer.encode_plus(sent, max_length=max_seq_length, pad_to_max_length=True, return_tensors='pt')
        input_ids_i = encoded["input_ids"]
        attention_mask_i = encoded["attention_mask"]
        
        entity_ids_i = torch.zeros(max_seq_length)
        
        # Can happen for long sentences when entity markers go out of boundary, ignore such cases
        try:
            e1_start, e1_end = (input_ids_i[0] == tokenizer.vocab[e1_tok]).nonzero().flatten()
        except:
            return []
        
        if entity_start:
            entity_ids_i[e1_start] = 1
        else:
            entity_ids_i[e1_start+1:e1_end] = 1
        
        try:
            e2_start, e2_end = (input_ids_i[0] == tokenizer.vocab[e2_tok]).nonzero().flatten()
        except:
            return []
        
        if entity_start:
            entity_ids_i[e2_start] = 2
        else:   
            entity_ids_i[e2_start+1:e2_end] = 2
        
        input_ids.append(input_ids_i)
        entity_ids.append(entity_ids_i.unsqueeze(0))
        attention_mask.append(attention_mask_i)
    
    # Happened once -- somehow?!
    if len(input_ids) == 0:
        return []
    
    group = (entity2idx[src], entity2idx[tgt])
    
    features = [dict(
        input_ids = torch.cat(input_ids),
        entity_ids = torch.cat(entity_ids),
        attention_mask = torch.cat(attention_mask),
        label = relation2idx[relation],
        group = group,
    ),]
    if config.expand_rels or not config.k_tag:
        # 0 = src "before" tgt; 1 = tgt "before" src
        rel_dir = 0 if jsonl["reldir"] == 1 else 1
        features[0]["rel_dir"] = rel_dir
    
    return features


def load_tokenizer(model_dir, do_lower_case=False):
    return BertTokenizer.from_pretrained(config.pretrained_model_dir, do_lower_case=do_lower_case)


def create_features(jsonl_fname, tokenizer, output_fname, entity2idx, relation2idx,
                    max_seq_length=128, e1_tok="$", e2_tok="^", entity_start=False):
    jr = list(iter(JsonlReader(jsonl_fname)))
    features = list()
    serial = False
    
    if serial:
        for idx, jsonl in enumerate(jr):
            if idx % 10000 == 0 and idx != 0:
                logger.info("Created {} features".format(idx))
            
            features.extend(tokenize_jsonl(
                jsonl, tokenizer, entity2idx, relation2idx,
                max_seq_length, e1_tok, e2_tok, entity_start
            ))
    else:
        func = functools.partial(
            tokenize_jsonl, tokenizer=tokenizer, entity2idx=entity2idx, 
            relation2idx=relation2idx, max_seq_length=max_seq_length, 
            e1_tok=e1_tok, e2_tok=e2_tok, entity_start=entity_start
        )
        with ProcessPoolExecutor(max_workers=4) as executor:
            for idx, features_idx in enumerate(executor.map(func, jr, chunksize=500)):
                if idx % 10000 == 0 and idx != 0:
                    logger.info("Created {} features".format(idx))
                if features_idx is None:
                    continue
                features.extend(copy.deepcopy(features_idx))
    
    torch.save(features, output_fname)


if __name__=="__main__":
    tokenizer = load_tokenizer(config.pretrained_model_dir, config.do_lower_case)
    entity2idx = read_entities(config.entities_file)
    relation2idx = read_relations(config.relations_file, with_dir=config.expand_rels)
    files = [
        (config.train_file, config.train_feats_file),
        (config.dev_file, config.dev_feats_file),
        (config.test_file, config.test_feats_file)
    ]
    for input_fname, output_fname in files:
        logger.info("Creating features for input `{}` ...".format(input_fname))
        create_features(
            input_fname, tokenizer, output_fname, entity2idx,
            relation2idx, config.max_seq_length, 
            entity_start=not config.entity_pool
        )
        logger.info("Saved features at `{}` ...".format(output_fname))
