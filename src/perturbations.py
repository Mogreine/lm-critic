import os
import math
import pickle
import random
from typing import List

import editdistance
from numpy.random import choice as npchoice
from collections import defaultdict

from definitions import ROOT_PATH
from src.tokenizer import TextPostprocessor

VERBS = pickle.load(open(f"{ROOT_PATH}/verbs.p", "rb"))
COMMON_INSERTS = set(pickle.load(open(f"{ROOT_PATH}/common_inserts.p", "rb")))  # common inserts *to fix a sent*
COMMON_DELETES = pickle.load(open(f"{ROOT_PATH}/common_deletes.p", "rb"))  # common deletes *to fix a sent*
_COMMON_REPLACES = pickle.load(open(f"{ROOT_PATH}/common_replaces.p", "rb"))  # common replacements *to error a sent*


COMMON_REPLACES = {}
for src in _COMMON_REPLACES:
    for tgt in _COMMON_REPLACES[src]:
        if (src == "'re" and tgt == "are") or (tgt == "'re" and src == "are"):
            continue
        ED = editdistance.eval(tgt, src)
        if ED > 2:
            continue
        longer = max(len(src), len(tgt))
        if float(ED) / longer >= 0.5:
            continue
        if tgt not in COMMON_REPLACES:
            COMMON_REPLACES[tgt] = {}
        COMMON_REPLACES[tgt][src] = _COMMON_REPLACES[src][tgt]


class WordLevelPerturbator:
    def _insert(self, sentence: str) -> str:
        """Insert a commonly deleted word."""
        sentence_tokenized = self.__tokenize(sentence)
        if len(sentence_tokenized) > 0:
            insertable = list(range(len(sentence_tokenized)))
            index = random.choice(insertable)
            plist = list(COMMON_DELETES.values())
            plistsum = sum(plist)
            plist = [x / plistsum for x in plist]
            # Choose a word
            ins_word = npchoice(list(COMMON_DELETES.keys()), p=plist)
            sentence_tokenized.insert(index, ins_word)
        return " ".join(sentence_tokenized)

    def _mod_verb(self, sentence: str, redir=True) -> str:
        sentence_tokenized = self.__tokenize(sentence)
        if len(sentence_tokenized) > 0:
            verbs = [i for i, w in enumerate(sentence_tokenized) if w in VERBS]
            if not verbs:
                if redir:
                    return self._replace(sentence, redir=False)
                return sentence
            index = random.choice(verbs)
            word = sentence_tokenized[index]
            if not VERBS[word]:
                return sentence
            repl = random.choice(VERBS[word])
            sentence_tokenized[index] = repl
        return " ".join(sentence_tokenized)

    def _delete(self, sentence: str) -> str:
        """Delete a commonly inserted word."""
        sentence_tokenized = self.__tokenize(sentence)
        if len(sentence_tokenized) > 1:
            toks_len = len(sentence_tokenized)
            toks = sentence_tokenized
            deletable = [i for i, w in enumerate(toks) if w in COMMON_INSERTS]
            if not deletable:
                return sentence
            index = random.choice(deletable)
            del sentence_tokenized[index]
        return " ".join(sentence_tokenized)

    def __tokenize(self, sentence: str):
        return sentence.split()

    def _replace(self, sentence: str, redir=True) -> str:
        sentence_tokenized = self.__tokenize(sentence)
        if len(sentence_tokenized) > 0:
            deletable = [i for i, w in enumerate(sentence_tokenized) if (w in COMMON_REPLACES)]
            if not deletable:
                if redir:
                    return self._mod_verb(sentence, redir=False)
                return sentence
            index = random.choice(deletable)
            word = sentence_tokenized[index]
            if not COMMON_REPLACES[word]:
                return sentence
            # Normalize probabilities
            plist = list(COMMON_REPLACES[word].values())
            plistsum = sum(plist)
            plist = [x / plistsum for x in plist]
            # Choose a word
            repl = npchoice(list(COMMON_REPLACES[word].keys()), p=plist)
            sentence_tokenized[index] = repl
        return " ".join(sentence_tokenized)

    def perturb(self, sentence: str) -> str:
        count = 1
        for x in range(count):
            perturb_probs = [0.30, 0.30, 0.30, 0.10]
            perturb_fun = npchoice([self._insert, self._mod_verb, self._replace, self._delete], p=perturb_probs)
            sentence = perturb_fun(sentence.split())
        return sentence

    def get_local_neighbors_word_level(self, sent_toked, max_n_samples=500):
        """ sent_toked is tokenized by spacy """
        n_samples = min(len(sent_toked) * 20, max_n_samples)
        original_sentence = " ".join(sent_toked)
        original_sentence_detokenized = TextPostprocessor.detokenize_sent(original_sentence)
        sent_perturbations = set()
        for _ in range(500):
            sent_perturbed = self.perturb(original_sentence)
            if sent_perturbed != original_sentence:
                sent_perturbed_detok = TextPostprocessor.detokenize_sent(sent_perturbed)
                sent_perturbations.add(sent_perturbed_detok)
            if len(sent_perturbations) == n_samples:
                break
        assert len(sent_perturbations) <= max_n_samples
        return sent_perturbations, original_sentence_detokenized
