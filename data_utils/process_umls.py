# -*- coding: utf-8 -*-

import logging
import collections
import nltk
import time
import itertools
import pickle
import config

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def iter_from_mrrel(mrrel_file, ro_only=False):
    """Reads UMLS relation triples file MRREL.RRF.
    
    Use ``ro_only`` to consider relations of "RO" semantic type only.
    50813206 lines in UMLS2019.
    
    For details on each column, please check:
    https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.related_concepts_file_mrrel_rrf/?report=objectonly
    
    """
    with open(mrrel_file) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            
            # Each line is as such: C0012792|A24166664|SCUI|RO|C0026827|A0088733|SCUI|induced_by|R176819430||MED-RT|MED-RT||N|N||
            line = line.split("|")
            
            # Consider relations of 'RO' type only
            if line[3] != "RO" and ro_only:
                continue
            
            e1_id = line[0]
            e2_id = line[4]
            rel_id = line[8]
            rel_text = line[7].strip()
            
            # considering relations with textual descriptions only
            if not rel_text:
                continue
            
            yield e1_id, (rel_id, rel_text), e2_id


def iter_from_mrconso(mrconso_file, en_only=False):
    """Reads UMLS concept names file MRCONSO.RRF.
    
    Use ``en_only`` to read English concepts only.
    11743183 lines for UMLS2019.
    
    For details on each column, please check:
    https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/?report=objectonly
    
    """
    with open(mrconso_file) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            
            # Each line is as such:  C0000005|ENG|P|L0000005|PF|S0007492|Y|A26634265||M0019694|D012711|MSH|PEP|D012711|(131)I-Macroaggregated Albumin|0|N|256|
            line = line.split("|")
            
            # Consider en only
            if line[1] != "ENG" and en_only:
                continue
            
            e_id = line[0]
            e_text = line[-5].strip()
            
            if not e_text:
                continue
            
            yield e_id, e_text


class UMLSVocab:
    """Class to hold UMLS entities, relations and their triples.
    
    """
    def __init__(self, mrrel_file, mrconso_file, en_only=True, ro_only=True):
        self.mrrel_file = mrrel_file
        self.mrconso_file = mrconso_file
        self.en_only = en_only
        self.ro_only = ro_only
    
    def build(self):
        """Parses UMLS MRREL.RRF and MRCONSO.RRF files to build mappings between
        entities, their texts and relations.
        
        """
        self.entity_text_to_cuis = collections.defaultdict(set)
        self.cui_to_entity_texts = collections.defaultdict(set)
        
        logger.info("Reading `{}` for UMLS concepts ...".format(self.mrconso_file))
        for e_cui, e_text in iter_from_mrconso(self.mrconso_file, en_only=self.en_only):
            # Ignore entities with char len = 2
            if len(e_text) <= 2:
                continue
            if e_text.lower() in STOPWORDS:
                continue
            self.entity_text_to_cuis[e_text].add(e_cui)
            self.cui_to_entity_texts[e_cui].add(e_text)
        
        logger.info("Collected {} unique CUIs and {} unique entities texts.".format(len(self.cui_to_entity_texts), len(self.entity_text_to_cuis)))
        self.relation_text_to_groups = collections.defaultdict(set)
        
        logger.info("Reading `{}` for UMLS relations triples ...".format(self.mrrel_file))
        for es_cui, (rel_id, rel_text), eo_cui in iter_from_mrrel(self.mrrel_file, ro_only=self.ro_only):
            self.relation_text_to_groups[rel_text].add((es_cui, eo_cui))
        
        all_groups = set()
        num_of_triples = 0
        for groups in self.relation_text_to_groups.values():
            all_groups.update(groups)
            num_of_triples += len(groups)
        num_of_groups = len(all_groups)
        
        logger.info("Collected {} unique relation texts.".format(len(self.relation_text_to_groups)))
        logger.info("Collected {} triples with {} unique groups.".format(num_of_triples, num_of_groups))
    
    def save(self, fname):
        args = (self.mrrel_file, self.mrconso_file,)
        kwargs = {"en_only": self.en_only, "ro_only": self.ro_only}
        data = {
            "entity_text_to_cuis": self.entity_text_to_cuis,
            "cui_to_entity_texts": self.cui_to_entity_texts,
            "relation_text_to_groups": self.relation_text_to_groups
        }
        save_data = (args, kwargs, data)
        with open(fname, "wb") as wf:
            pickle.dump(save_data, wf)
    
    @staticmethod
    def load(fname):
        with open(fname, "rb") as rf:
            load_data = pickle.load(rf)
        uv = UMLSVocab(load_data[0][0], load_data[0][1], **load_data[1])
        uv.entity_text_to_cuis = load_data[2]["entity_text_to_cuis"]
        uv.cui_to_entity_texts = load_data[2]["cui_to_entity_texts"]
        uv.relation_text_to_groups = load_data[2]["relation_text_to_groups"]
        return uv


if __name__=="__main__":
    uv = UMLSVocab(config.mrrel_file, config.mrconso_file)
    uv.build()
    # Save the UMLS vocab
    logger.info("Saving UMLS vocab object at {} ...".format(config.umls_vocab_file))
    uv.save(config.umls_vocab_file) 
