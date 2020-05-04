# -*- coding:utf-8 -*-

import json


class JsonlReader:
    
    def __init__(self, fname):
        self.fname = fname
    
    def __iter__(self):
        with open(self.fname, encoding="utf-8", errors="ignore") as rf:
            for jsonl in rf:
                jsonl = jsonl.strip()
                if not jsonl:
                    continue
                yield json.loads(jsonl)


class TriplesReader:
    
    def __init__(self, fname):
        self.fname = fname
    
    def __iter__(self):
        with open(self.fname, encoding="utf-8", errors="ignore") as rf:
            for tsvl in rf:
                tsvl = tsvl.strip()
                if not tsvl:
                    continue
                yield tsvl.split("\t")


def read_relations(relations_file, with_dir=False):
    relation2idx = dict()
    idx = 0
    with open(relations_file) as rf:
        for relation in rf:
            relation = relation.strip()
            if not relation:
                continue
            if with_dir and relation != "NA":
                relation2idx[relation+"(e1,e2)"] = idx
                idx += 1
                relation2idx[relation+"(e2,e1)"] = idx
                idx += 1
            else:
                relation2idx[relation] = idx
                idx += 1
    return relation2idx


def read_entities(entities_file):
    entity2idx = dict()
    idx = 0
    with open(entities_file) as rf:
        for entity in rf:
            entity = entity.strip()
            if not entity:
                continue
            entity2idx[entity] = idx
            idx += 1
    return entity2idx
