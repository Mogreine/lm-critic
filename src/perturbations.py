import json
import os
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from typing import List

import editdistance
import numpy as np
from numpy.random import choice as npchoice
from random import sample

from definitions import ROOT_PATH
from src.edit_dist_utils import get_all_edit_dist_one
from src.tokenizer import TextPostprocessor

VERBS = pickle.load(open(f"{ROOT_PATH}/artifacts/verbs.p", "rb"))
COMMON_INSERTS = set(
    pickle.load(open(f"{ROOT_PATH}/artifacts/common_inserts.p", "rb"))
)  # common inserts *to fix a sent*
COMMON_DELETES = pickle.load(open(f"{ROOT_PATH}/artifacts/common_deletes.p", "rb"))  # common deletes *to fix a sent*
_COMMON_REPLACES = pickle.load(
    open(f"{ROOT_PATH}/artifacts/common_replaces.p", "rb")
)  # common replacements *to error a sent*


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


VERBS_refine = defaultdict(list)
for src in VERBS:
    for tgt in VERBS[src]:
        ED = editdistance.eval(tgt, src)
        if ED > 2:
            continue
        longer = max(len(src), len(tgt))
        if float(ED) / longer >= 0.5:
            continue
        VERBS_refine[src].append(tgt)


class WordLevelPerturbatorBase:
    def __init__(self):
        self.verbs = VERBS

    def _filter_tokens_to_delete(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError()

    def _filter_tokens_to_replace(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError()

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

    def perturb(self, sentence: str) -> str:
        count = 1
        for x in range(count):
            perturb_probs = [0.30, 0.30, 0.30, 0.10]
            perturb_fun = npchoice([self._insert, self._mod_verb, self._replace, self._delete], p=perturb_probs)
            sentence = perturb_fun(sentence)
        return sentence

    def __tokenize(self, sentence: str):
        return sentence.split()

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
            verbs = [i for i, w in enumerate(sentence_tokenized) if w in self.verbs]
            if not verbs:
                if redir:
                    return self._replace(sentence, redir=False)
                return sentence
            index = random.choice(verbs)
            word = sentence_tokenized[index]
            if not self.verbs[word]:
                return sentence
            repl = random.choice(self.verbs[word])
            sentence_tokenized[index] = repl
        return " ".join(sentence_tokenized)

    def _delete(self, sentence: str) -> str:
        """Delete a commonly inserted word."""
        sentence_tokenized = self.__tokenize(sentence)
        if len(sentence_tokenized) > 1:
            toks_len = len(sentence_tokenized)
            toks = sentence_tokenized
            deletable = self._filter_tokens_to_delete(toks)
            if not deletable:
                return sentence
            index = random.choice(deletable)
            del sentence_tokenized[index]
        return " ".join(sentence_tokenized)

    def _replace(self, sentence: str, redir=True) -> str:
        sentence_tokenized = self.__tokenize(sentence)
        if len(sentence_tokenized) > 0:
            deletable = self._filter_tokens_to_replace(sentence_tokenized)

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


class WordLevelPerturbator(WordLevelPerturbatorBase):
    def _filter_tokens_to_delete(self, tokens: List[str]) -> List[int]:
        return [i for i, tok in enumerate(tokens) if tok in COMMON_INSERTS]

    def _filter_tokens_to_replace(self, tokens: List[str]) -> List[int]:
        return [i for i, tok in enumerate(tokens) if tok in COMMON_REPLACES]


class WordLevelPerturbatorRefined(WordLevelPerturbatorBase):
    def __init__(self):
        super().__init__()
        self.verbs = VERBS_refine

    def _filter_tokens_to_delete(self, tokens: List[str]) -> List[int]:
        return [
            i
            for i, tok in enumerate(tokens)
            if tok in COMMON_INSERTS and i > 0 and tokens[i - 1].lower() == tokens[i].lower()
        ]

    def _filter_tokens_to_replace(self, tokens: List[str]) -> List[int]:
        return [i for i, tok in enumerate(tokens) if tok in COMMON_REPLACES and tok.lower() not in {"not", "n't"}]


class CharLevelPerturbator:
    def __init__(self, attack_type="ed1"):
        self.cache = {}  # {word: {0: set(), 1: set(),.. }, ..} #0=swap, 1=substitute, 2=delete, 3=insert
        self.n_types = 5
        self.attack_type = attack_type
        self.common_typo = json.load(open(f"{ROOT_PATH}/artifacts/common_typo.json"))

    def __tokenize(self, sent):
        toks = []
        word_idxs = []
        for idx, match in enumerate(re.finditer(r"([a-zA-Z]+)|([0-9]+)|.", sent)):
            tok = match.group(0)
            toks.append(tok)
            if len(tok) > 2 and tok.isalpha() and (tok[0].islower()):
                word_idxs.append(idx)
        return toks, word_idxs

    def __detokenize(self, toks):
        return "".join(toks)

    def sample_perturbations(self, word, n_samples, types):
        if types is None:
            type_list = list(range(4)) * (n_samples // 4) + list(
                np.random.choice(self.n_types, n_samples % self.n_types, replace=False)
            )
        else:
            type_list = [sample(types, 1)[0] for _ in range(n_samples)]
        type_count = Counter(type_list)
        perturbations = set()
        for type in type_count:
            if type not in self.cache[word]:
                continue
            if len(self.cache[word][type]) >= type_count[type]:
                perturbations.update(set(sample(self.cache[word][type], type_count[type])))
            else:
                perturbations.update(self.cache[word][type])
        return perturbations

    #
    def get_perturbations(self, word, n_samples, types=None):
        if word not in self.cache:
            self.cache[word] = {}
            if word[0].islower():
                for type in range(4):
                    self.cache[word][type] = get_all_edit_dist_one(word, 10 ** type)
                if word in self.common_typo:
                    self.cache[word][4] = set(self.common_typo[word])
            elif word[0].isupper():
                if word in self.common_typo:
                    self.cache[word][4] = set(self.common_typo[word])
        if self.attack_type == "ed1":
            perturbations = self.sample_perturbations(word, n_samples, types)
        else:
            raise NotImplementedError("Attack type: {} not implemented yet".format(self.attack_type))
        return perturbations

    #
    def name(self):
        return "RandomPerturbationAttack"

    def get_local_neighbors_char_level(self, sent, max_n_samples=500) -> set:
        words, word_idxs = self.__tokenize(sent)
        n_samples = min(len(word_idxs) * 20, max_n_samples)
        sent_perturbations = set()
        if len(word_idxs) == 0:
            return sent_perturbations
        for _ in range(500):
            word_idx = sample(word_idxs, 1)[0]
            words_cp = words[:]
            word_perturbations = list(self.get_perturbations(words_cp[word_idx], n_samples=1))
            if len(word_perturbations) > 0:
                words_cp[word_idx] = word_perturbations[0]
                sent_perturbed = self.__detokenize(words_cp)
                if sent_perturbed != sent:
                    sent_perturbations.add(sent_perturbed)
            if len(sent_perturbations) == n_samples:
                break
        # Adding common typos such as 's'
        for word_idx in word_idxs:
            words_cp = words[:]
            word = words_cp[word_idx]
            if len(word) > 2 and word[0].islower():
                words_cp[word_idx] = word + "s"
                sent_perturbed = self.__detokenize(words_cp)
                if sent_perturbed != sent:
                    sent_perturbations.add(sent_perturbed)
                words_cp[word_idx] = word[:-1]
                sent_perturbed = self.__detokenize(words_cp)
                if sent_perturbed != sent:
                    sent_perturbations.add(sent_perturbed)
        if len(sent_perturbations) > max_n_samples:
            sent_perturbations = list(sent_perturbations)
            np.random.shuffle(sent_perturbations)
            sent_perturbations = set(sent_perturbations[:max_n_samples])
        return sent_perturbations
