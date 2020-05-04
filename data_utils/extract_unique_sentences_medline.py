# -*- coding: utf-8 -*-

import logging
import nltk
import hashlib
import json
import time
import config

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MEDLINESents:
    
    def __init__(self, medline_abstracts, output_fname, lowercase=False):
        self.medline_abstracts = medline_abstracts
        self.output_fname = output_fname
        self.sent_tok = nltk.data.load("tokenizers/punkt/english.pickle").tokenize
        self.lowercase = lowercase
        self.n = 0
        self.d = 0
    
    def process_abstract(self, doc):
        # Strip starting b' or b" and ending ' or "
        if (doc[:2] == "b'" and doc[-1] == "'") or (doc[:2] == 'b"' and doc[-1] == '"'):
            doc = doc[2:-1]
        
        # Sentence tokenization
        for sent in self.sent_tok(doc):
            if self.lowercase:
                sent = sent.lower()
            shash = hashlib.sha256(sent.encode("utf-8")).hexdigest()
            if not shash in self.hash_set:
                self.hash_set.add(shash)
                self.n += 1
                yield sent
            else:
                self.d += 1
    
    def extract_unique_sentences(self):
        self.hash_set = set()
        
        logger.info("Extracting unique sentences from `{}` ...".format(self.medline_abstracts))
        with open(self.medline_abstracts, encoding="utf-8", errors="ignore") as rf, open(self.output_fname, "w") as wf:
            for idx, abstract in enumerate(rf):
                if idx % 100000 == 0 and idx != 0:
                    logger.info("Read %d documents : extracted %d unique sentences (dupes = %d)" % (idx, self.n, self.d))
                abstract = abstract.strip()
                if not abstract:
                    continue
                for sent in self.process_abstract(abstract):
                    wf.write(sent + "\n")
        
        del self.hash_set


if __name__=="__main__":
    ms = MEDLINESents(config.medline_file, config.medline_unique_sents_file)
    t = time.time()
    ms.extract_unique_sentences()
    t = (time.time() - t) // 60
    logger.info("Took {} mins!".format(t))
