# -*- coding: utf-8 -*-

import os
import torch

DATA_DIR = "data"
UMLS_DIR = os.path.join("data", "UMLS")
MEDLINE_DIR = os.path.join("data", "MEDLINE")
mrrel_file = os.path.join(UMLS_DIR, "MRREL.RRF")
mrconso_file = os.path.join(UMLS_DIR, "MRCONSO.RRF")
medline_file = os.path.join(MEDLINE_DIR, "medline_abs.txt")
medline_unique_sents_file = os.path.join(MEDLINE_DIR, "medline_unique_sentences.txt")
medline_linked_sents_file = os.path.join(MEDLINE_DIR, "umls_linked_sentences.jsonl")
groups_linked_sents_file = os.path.join(MEDLINE_DIR, "linked_sentences_to_groups.jsonl")
umls_vocab_file = os.path.join("data", "umls_vocab.pkl")

# Main configurations
entity_pool = True # True to use average of sub-words, False for only first sub-token (can only be used with special tokens)
k_tag = True # K-Tag or S-tag scheme (can only be used with special tokens)
expand_rels = False and not k_tag # To expand relations as (e1,e2), (e2,e1) (can only be used with S-tag)
bag_attn = False

config_name = [
    ("avg_entpool." if entity_pool else "first_entpool.") if entity_pool is not None else "",
    ("k_tag." if k_tag else "s_tag.") if k_tag is not None else "",
    "expand_rels." if expand_rels else ""
]
config_name = "".join(config_name)[:-1]

DATA_DIR = "data"
SAVE_DIR = os.path.join(DATA_DIR, config_name)
FEATURES_DIR = os.path.join(SAVE_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

triples_file = os.path.join(SAVE_DIR, "triples.tsv")
entities_file = os.path.join(SAVE_DIR, "entities.txt")
relations_file = os.path.join(SAVE_DIR, "relations.txt")
rel2id_file = os.path.join(SAVE_DIR, "rel2id.json")

train_triples_file = os.path.join(SAVE_DIR, "train_triples.tsv")
dev_triples_file = os.path.join(SAVE_DIR, "dev_triples.tsv")
test_triples_file = os.path.join(SAVE_DIR, "test_triples.tsv")

train_file = os.path.join(SAVE_DIR, "pubmed_train.txt")
dev_file = os.path.join(SAVE_DIR, "pubmed_dev.txt")
test_file = os.path.join(SAVE_DIR, "pubmed_test.txt")

# Entity linking options
case_sensitive_linker = True
min_sent_char_len_linker = 32
max_sent_char_len_linker = 256
min_rel_group = 10
max_rel_group = 1500

bag_size = 16
max_seq_length = 128

SEED = 2019

# Models
pretrained_model_dir = "path/to/pretrained/biobert/biobert_v1.1_pubmed" # OR any other BERT like model
do_lower_case = False

# Features files
train_feats_file = os.path.join(FEATURES_DIR, "train.pt")
dev_feats_file = os.path.join(FEATURES_DIR, "dev.pt")
test_feats_file = os.path.join(FEATURES_DIR, "test.pt")

# Training args
cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
n_gpu = 0 if not cuda else torch.cuda.device_count()

per_gpu_train_batch_size = 2 # 4
train_batch_size = 1
per_gpu_eval_batch_size = 16 # 24
eval_batch_size = 1
gradient_accumulation_steps = 1

num_train_epochs = 3
learning_rate = 2e-5

adam_epsilon = 1e-8
warmup_percent = 0.01
max_grad_norm = 1.0
weight_decay = 0.

logging_steps = 5000
evaluate_during_training = True
save_steps = 5000
if bag_attn:
    config_name += ".bag_attn"
output_dir = os.path.join("models", config_name)
os.makedirs(output_dir, exist_ok=True)

do_train = True
do_eval = True
test_ckpt = output_dir
checkpoint = None
max_steps = 125000
