
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
"""

Script adopted from:
    https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

"""

import json
import logging
import os
import random
import pickle

import numpy as np
import torch
import config

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

from transformers import BertConfig
from tensorboardX import SummaryWriter
from model import BertForDistantRE
from sklearn import metrics
from data_utils import read_relations, read_entities
from data_utils import TriplesReader as read_triples

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed():
    seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def add_logging_handlers(logger, dir_name):
    log_file = os.path.join(dir_name, "run.log")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', '%m/%d/%Y %H:%M:%S'
    ))
    logger.addHandler(fh)


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)
    
    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


def train(train_dataset, model):
    tb_writer = SummaryWriter()
    train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    t_total = min(len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs, 75000)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters , lr=config.learning_rate, eps=config.adam_epsilon)
    warmup_steps = int(config.warmup_percent * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    
    if config.checkpoint is not None:
        model.load_state_dict(torch.load(config.checkpoint + "/pytorch_model.bin"))
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(config.checkpoint + "/optimizer.pt"))
        scheduler.load_state_dict(torch.load(config.checkpoint + "/scheduler.pt"))
        _, ckpt_global = os.path.split(config.checkpoint)[1].split("-")
        ckpt_global = int(ckpt_global)
    else:
        ckpt_global = 0
    
    # multi-gpu training (should be after apex fp16 initialization)
    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", config.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        config.train_batch_size * config.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = ckpt_global
    tr_loss, logging_loss, auc, best_auc = 0.0, 0.0, 0.0, 0.0
    best_results = dict()
    rel2idx = read_relations(config.relations_file, config.expand_rels)
    na_idx = rel2idx["NA"]
    losses, accs, pos_accs = list(), list(), list()
    model.zero_grad()
    train_iterator = trange(0, int(config.num_train_epochs), desc="Epoch",)
    set_seed()
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_pos_acc = AverageMeter()
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(config.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "entity_ids": batch[1],
                "attention_mask": batch[2],
                "labels": batch[4],
                "is_train": True
            }
            outputs = model(**inputs)
            loss = outputs[0]  
            logits = outputs[1]
            
            # Train results
            preds = torch.argmax(torch.nn.Softmax(-1)(logits), -1)
            acc = float((preds == inputs["labels"]).long().sum()) / inputs["labels"].size(0)
            pos_total = (inputs["labels"] != na_idx).long().sum()
            pos_correct = ((preds == inputs["labels"]).long() * (inputs["labels"] != na_idx).long()).sum()
            if pos_total > 0:
                pos_acc = float(pos_correct) / float(pos_total)
            else:
                pos_acc = 0
            
            if config.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            # Log
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            #
            if global_step % 100 == 0:
                logger.info(" tr_loss = %s", str(avg_loss.avg))
                logger.info(" tr_accuracy = %s", str(avg_acc.avg))
                logger.info(" tr_pos_accuracy = %s", str(avg_pos_acc.avg))
            
            loss.backward()
            
            tr_loss += loss.item()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                    logs = {}
                    if config.evaluate_during_training:
                        results = evaluate(model, "dev")
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            if key == "R" or key == "P":
                                continue
                            logs[eval_key] = value
                        if results["AUC"] > best_auc:
                            logger.info("  ***  Best ckpt and saved  ***  ")
                            best_results = results
                            best_auc = results["AUC"]
                            
                            # Save model checkpoint
                            output_dir = os.path.join(config.output_dir, "{}-best-{}".format(global_step, best_auc))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (model.module if hasattr(model, "module") else model)
                            model_to_save.save_pretrained(output_dir)
                            save_eval_results(results, output_dir, "dev")
                    
                    loss_scalar = (tr_loss - logging_loss) / config.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss
                    
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))
                
                losses.append(loss.item())
                accs.append(acc)
                pos_accs.append(pos_acc)
                
                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(config.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (model.module if hasattr(model, "module") else model)
                    model_to_save.save_pretrained(output_dir)
                    
                    logger.info("Saving model checkpoint to %s", output_dir)
                    
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
            
            if config.max_steps > 0 and global_step > config.max_steps:
                epoch_iterator.close()
                break
        
        if config.max_steps > 0 and global_step > config.max_steps:
            train_iterator.close()
            break
    
    results = evaluate(model, set_type="dev", prefix="final-{}".format(global_step))
    if results["AUC"] > best_auc:
        best_results = results
        best_auc = results["AUC"]
    
    # Save model checkpoint
    output_dir = os.path.join(config.output_dir, "final-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(output_dir)
    tb_writer.close()
    tr_data = (losses, accs, pos_accs)
    logger.info("***** Best eval AUC : {} *****".format(best_auc))
    logger.info("***** Best dev results *****")
    for key in sorted(best_results.keys()):
        logger.info("  %s = %s", key, str(best_results[key]))
    
    return global_step, tr_loss / global_step, tr_data


def evaluate(model, set_type="dev", prefix=""):
    results = {}
    
    eval_output_dir = config.output_dir
    eval_dataset = load_dataset(set_type)
    
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    
    config.eval_batch_size = config.per_gpu_eval_batch_size * max(1, config.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.eval_batch_size)
    
    # multi-gpu eval
    if config.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", config.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    eval_logits = list()
    eval_labels = list()
    eval_groups = list()
    eval_dirs = list()
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)
        
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "entity_ids": batch[1],
                "attention_mask": batch[2],
                "labels": batch[4],
                "is_train": False
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        
        eval_labels.append(inputs["labels"].detach().cpu())
        eval_logits.append(logits.detach().cpu())
        eval_groups.append(batch[3].detach().cpu()) # groups
        if config.expand_rels or not config.k_tag:
            eval_dirs.append(batch[5].detach().cpu()) # relation dirs in text
    
    eval_loss = eval_loss / nb_eval_steps
    eval_labels = torch.cat(eval_labels) # B
    eval_logits = torch.cat(eval_logits) # B x C
    eval_groups = torch.cat(eval_groups) # B x 2
    if config.expand_rels or not config.k_tag:
        eval_dirs = torch.cat(eval_dirs)
    else:
        eval_dirs = None
    
    results["loss"] = eval_loss
    result = compute_metrics(eval_logits, eval_labels, eval_groups, set_type, eval_dirs)
    results.update(result)
    
    eval_dir = os.path.join(eval_output_dir, prefix)
    save_eval_results(results, eval_dir, set_type, prefix)
    
    return results


def save_eval_results(results, eval_dir, set_type, prefix=""):
    os.makedirs(eval_dir, exist_ok=True)
    output_eval_file = os.path.join(eval_dir, "eval_results.txt")
    
    with open(output_eval_file, "w") as wf:
        logger.info("***** {} results {} *****".format(set_type, prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            wf.write("%s = %s\n" % (key, str(results[key])))


def load_dataset(set_type):
    
    if set_type == "train":
        features_file = config.train_feats_file
    elif set_type == "dev":
        features_file = config.dev_feats_file
    else:
        features_file = config.test_feats_file
    
    logger.info("Loading features from cached file %s", features_file)
    features = torch.load(features_file)
    
    all_input_ids = torch.cat([f["input_ids"].unsqueeze(0) for f in features]).long()
    all_entity_ids = torch.cat([f["entity_ids"].unsqueeze(0) for f in features]).long()
    all_attention_mask = torch.cat([f["attention_mask"].unsqueeze(0) for f in features]).long()
    all_groups = torch.cat([torch.tensor(f["group"]).unsqueeze(0) for f in features]).long()
    all_labels = torch.tensor([f["label"] for f in features]).long()
    if config.expand_rels or not config.k_tag:
        all_dirs = torch.tensor([f["rel_dir"] for f in features]).long()
        dataset = TensorDataset(all_input_ids, all_entity_ids, all_attention_mask, all_groups, all_labels, all_dirs) 
    else:
        dataset = TensorDataset(all_input_ids, all_entity_ids, all_attention_mask, all_groups, all_labels) 
    
    return dataset

# cf. https://stackoverflow.com/a/480227
def non_dup_ordered_seq(seq):
    seen = set()
    seen_add = seen.add
    non_dup_seq = list()
    for item in seq:
        relation = item["relation"]
        src, tgt = item["src"], item["tgt"]
        triple = (src, relation, tgt)
        if not (triple in seen or seen_add(triple)):
            non_dup_seq.append(item)
    return non_dup_seq


def compute_metrics(logits, labels, groups, set_type, rel_dirs=None):
    
    # Read relation mappings 
    rel2idx = read_relations(config.relations_file, config.expand_rels)
    idx2rel = {v:k for k, v in rel2idx.items()}
    entity2idx = read_entities(config.entities_file)
    
    # Read triples
    triples = set()
    if set_type == "dev":
        triples_file = config.dev_triples_file
    else:
        triples_file = config.test_triples_file
    for src, rel, tgt in read_triples(triples_file):
        if rel != "NA":
            triples.add((entity2idx[src], rel, entity2idx[tgt]))
    
    # RE predictions
    probas = torch.nn.Softmax(-1)(logits)
    re_preds = list()
    for i in range(probas.size(0)):
        if config.expand_rels:
            group = groups[i]
            rdir = rel_dirs[i].item()
            if rdir == 0:
                src, tgt = group[0].item(), group[1].item()
            else:
                src, tgt = group[1].item(), group[0].item()
        else:   
            group = groups[i]
            src, tgt = group[0].item(), group[1].item()
        for rel, rel_idx in rel2idx.items():
            if rel != "NA":
                re_preds.append({
                    "src": src, "tgt": tgt,
                    "relation": rel, 
                    "score": probas[i][rel_idx].item()
                })
    
    # Adopted from:
    # https://github.com/thunlp/OpenNRE/blob/master/opennre/framework/data_loader.py#L230
    
    sorted_re_preds = sorted(re_preds, key=lambda x: x["score"], reverse=True)
    sorted_re_preds = non_dup_ordered_seq(sorted_re_preds)
    P = list()
    R = list()
    correct = 0
    total = len(triples)
    
    for i, item in enumerate(sorted_re_preds):
        relation = item["relation"]
        src, tgt = item["src"], item["tgt"]
        if (src, relation, tgt) in triples:
            correct += 1
        P.append(float(correct) / float(i + 1))
        R.append(float(correct) / float(total))
    
    auc = metrics.auc(x=R, y=P)
    P = np.array(P)
    R = np.array(R) 
    f1 = (2 * P * R / (P + R + 1e-20)).max()
    avg_P = P.mean()
    P2k = sum(P[:2000]) / 2000
    P4k = sum(P[:4000]) / 4000
    P6k = sum(P[:6000]) / 6000
    
    results = {"P": P, "R": R, "P@2k": P2k, "P@4k": P4k, "P@6k": P6k, "F1": f1, "AUC": auc,
    "P@100": sum(P[:100])/100, "P@200": sum(P[:200])/200, "P@300": sum(P[:300])/300,
    "P@500": sum(P[:500])/500, "P@1000": sum(P[:1000])/1000}
    
    return results


def main():
    num_labels = len(read_relations(config.relations_file, config.expand_rels))
    num_ents = len(read_entities(config.entities_file))
    model = BertForDistantRE(BertConfig.from_pretrained(config.pretrained_model_dir), num_labels, bag_attn=config.bag_attn)
    model.to(config.device)
    
    add_logging_handlers(logger, config.output_dir)
    
    # Training
    if config.do_train:
        train_dataset = load_dataset("train")
        global_step, tr_loss, tr_data = train(train_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if config.do_train:
        # Create output directory if needed
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        
        logger.info("Saving model checkpoint to %s", config.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(config.output_dir)
        
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForDistantRE(BertConfig.from_pretrained(config.output_dir), num_labels, bag_attn=config.bag_attn)
        model.to(config.device)
    
    # Evaluation
    results = {}
    if config.do_eval:
        checkpoint = config.test_ckpt
        logger.info("Evaluate the checkpoint: %s", checkpoint)
        
        model = BertForDistantRE(BertConfig.from_pretrained(checkpoint), num_labels, bag_attn=config.bag_attn)
        model.load_state_dict(torch.load(checkpoint + "/pytorch_model.bin"))
        model.to(config.device)
        result = evaluate(model, "test", prefix="TEST")
        global_step = ""
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)
        with open(os.path.join(checkpoint, "pr_metrics.pkl"), "wb") as wf:
            pickle.dump(results, wf) 
    
    return results


if __name__=="__main__":
    main()
