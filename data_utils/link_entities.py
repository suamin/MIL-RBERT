# -*- coding: utf-8 -*-

import logging
import collections
import time
import json
import config
import pickle

from flashtext import KeywordProcessor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class ExactEntityLinking:
    
    def __init__(self, entities, case_sensitive=True):
        self.linker = KeywordProcessor(case_sensitive=case_sensitive)
        
        logger.info("Building Trie data structure with flashText for exact match entity linking (|E|={}) ...".format(len(entities)))
        
        t = time.time()
        self.linker.add_keywords_from_list(list(set(entities)))
        t = (time.time() - t) // 60
        
        logger.info("Took %d mins" % t)
    
    def link(self, text):
        spans = sorted([(start_span, end_span) for _, start_span, end_span in self.linker.extract_keywords(text, span_info=True)], key=lambda span: span[0])
        if not spans:
            return
        
        # Remove overlapping matches, if any
        filtered_spans = list()
        for i in range(1, len(spans)):
            span_prev, span_next = spans[i-1], spans[i]
            if span_prev[1] < span_next[0]:
                filtered_spans.append(spans[i])
        spans = filtered_spans[:]
        
        matches_texts = [text[s:e] for s, e in spans]
        # Check if any entity is present more than once, drop this sentence
        counts = collections.Counter(matches_texts)
        skip = False
        
        for _, count in counts.items():
            if count > 1:
                skip = True
                break
        if skip:
            return
        
        text2span = {matches_texts[i]:spans[i] for i in range(len(spans))}
        
        return text2span


def link_sentences(linker, sents_fname, output_fname):
    t = time.time()
    
    with open(sents_fname, encoding="utf-8", errors="ignore") as rf, open(output_fname, "w", encoding="utf-8", errors="ignore") as wf:
        for idx, sent in enumerate(rf):
            if idx % 1000000 == 0 and idx != 0:
                logger.info("Checked {} sentences for entity linking".format(idx))
            sent = sent.strip()
            if not sent:
                continue
            # Skip short or very long sentences
            if (len(sent) < config.min_sent_char_len_linker) or (len(sent) > config.max_sent_char_len_linker):
                continue
            text2span = linker.link(sent)
            if text2span is None:
                continue
            jdata = {"sent": sent, "matches": text2span}
            wf.write(json.dumps(jdata) + "\n")
    
    t = (time.time() - t) // 60
    logger.info("Took %d mins (%d sents / min)" % (t, (idx+1)/t))


if __name__=="__main__":
    with open(config.umls_vocab_file, "rb") as rf:
        uv = pickle.load(rf)
    linker = ExactEntityLinking(uv.entity_text_to_cuis.keys(), config.case_sensitive_linker)
    link_sentences(linker, config.medline_unique_sents_file, config.medline_linked_sents_file)
