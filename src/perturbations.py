import json
import os.path
import pickle
import random
import re
import editdistance
import numpy as np

from typing import List, Tuple, Set
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from random import sample

from definitions import ROOT_PATH
from src.utils import get_all_edit_dist_one
from src.tokenizer import TextPostprocessor


class WordLevelPerturbatorBase(ABC):
    def __init__(self):
        self._word_forms = None
        self._common_inserts = None
        self._common_deletes = None
        self._common_replaces = None
        self._verbs_refine = None
        self._common_replaces_refine = None
        self._max_iterations = 500
        self._load_common_modifications()

    @abstractmethod
    def _filter_tokens_to_delete(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError()

    @abstractmethod
    def _filter_tokens_to_replace(self, tokens: List[str]) -> List[int]:
        raise NotImplementedError()

    def _load_common_modifications(self):
        with open(os.path.join(ROOT_PATH, "data/verbs.p"), "rb") as file:
            self._word_forms = pickle.load(file)
        with open(os.path.join(ROOT_PATH, "data/common_inserts.p"), "rb") as file:
            self._common_inserts = set(pickle.load(file))
        with open(os.path.join(ROOT_PATH, "data/common_deletes.p"), "rb") as file:
            self._common_deletes = pickle.load(file)
        with open(os.path.join(ROOT_PATH, "data/common_replaces.p"), "rb") as file:
            self._common_replaces = pickle.load(file)

        self._common_replaces_refine = {}
        for src in self._common_replaces:
            for tgt in self._common_replaces[src]:
                if (src == "'re" and tgt == "are") or (tgt == "'re" and src == "are"):
                    continue
                edit_dist = editdistance.eval(tgt, src)
                if edit_dist > 2:
                    continue
                longer = max(len(src), len(tgt))
                if edit_dist / longer >= 0.5:
                    continue
                if tgt not in self._common_replaces_refine:
                    self._common_replaces_refine[tgt] = {}
                self._common_replaces_refine[tgt][src] = self._common_replaces[src][tgt]

    def get_local_neighbors(self, sentence_tokenized, max_n_samples=500):
        n_samples = min(len(sentence_tokenized) * 20, max_n_samples)
        original_sentence = " ".join(sentence_tokenized)
        original_sentence_detokenized = TextPostprocessor.detokenize_sent(original_sentence)
        sent_perturbations = set()

        for _ in range(self._max_iterations):
            sent_perturbed = self.perturb(original_sentence)
            if sent_perturbed != original_sentence:
                sent_perturbed_detok = TextPostprocessor.detokenize_sent(sent_perturbed)
                sent_perturbations.add(sent_perturbed_detok)
            if len(sent_perturbations) == n_samples:
                break

        assert len(sent_perturbations) <= max_n_samples
        return sent_perturbations, original_sentence_detokenized

    def perturb(self, sentence: str) -> str:
        perturb_probs = [0.30, 0.30, 0.30, 0.10]
        perturb_fun = np.random.choice([self._insert, self._mod_word, self._replace, self._delete], p=perturb_probs)
        sentence = perturb_fun(sentence)
        return sentence

    def _tokenize(self, sentence: str):
        return sentence.split()

    def _insert(self, sentence: str) -> str:
        """Insert a commonly deleted word."""
        sentence_tokenized = self._tokenize(sentence)
        if len(sentence_tokenized) > 0:
            insertable = list(range(len(sentence_tokenized)))
            index = random.choice(insertable)
            plist = list(self._common_deletes.values())
            plist = [x / sum(plist) for x in plist]

            # Choose a word
            ins_word = np.random.choice(list(self._common_deletes.keys()), p=plist)
            sentence_tokenized.insert(index, ins_word)

        return " ".join(sentence_tokenized)

    def _mod_word(self, sentence: str, redir: bool = True) -> str:
        """
        Tries to replace a word by its other forms (plural or singular in case of nouns and tense dependent in case of verbs).
        It samples the words uniformly.
        """
        sentence_tokenized = self._tokenize(sentence)

        if len(sentence_tokenized) > 0:
            verbs = [i for i, w in enumerate(sentence_tokenized) if w in self._word_forms]

            if not verbs:
                if redir:
                    return self._replace(sentence, redir=False)
                return sentence

            index = random.choice(verbs)
            word = sentence_tokenized[index]
            if not self._word_forms[word]:
                return sentence
            replacement = random.choice(self._word_forms[word])
            sentence_tokenized[index] = replacement

        return " ".join(sentence_tokenized)

    def _delete(self, sentence: str) -> str:
        """Delete a commonly inserted word."""
        sentence_tokenized = self._tokenize(sentence)

        if len(sentence_tokenized) > 1:
            toks = sentence_tokenized
            deletable = self._filter_tokens_to_delete(toks)
            if not deletable:
                return sentence
            index = random.choice(deletable)
            del sentence_tokenized[index]

        return " ".join(sentence_tokenized)

    def _replace(self, sentence: str, redir: bool = True) -> str:
        """
        Tries to replace a word by its common replacements.
        It samples the words according to the popularity of a replacement.
        """
        sentence_tokenized = self._tokenize(sentence)
        if len(sentence_tokenized) > 0:
            replaceable = self._filter_tokens_to_replace(sentence_tokenized)

            if not replaceable:
                if redir:
                    return self._mod_word(sentence, redir=False)
                return sentence

            index = random.choice(replaceable)
            word = sentence_tokenized[index]
            if not self._common_replaces_refine[word]:
                return sentence

            # Normalize probabilities
            plist = list(self._common_replaces_refine[word].values())
            plist = [x / sum(plist) for x in plist]

            # Choose a word
            replacement = np.random.choice(list(self._common_replaces_refine[word].keys()), p=plist)
            sentence_tokenized[index] = replacement

        return " ".join(sentence_tokenized)


class WordLevelPerturbator(WordLevelPerturbatorBase):
    def _filter_tokens_to_delete(self, tokens: List[str]) -> List[int]:
        return [i for i, tok in enumerate(tokens) if tok in self._common_inserts]

    def _filter_tokens_to_replace(self, tokens: List[str]) -> List[int]:
        return [i for i, tok in enumerate(tokens) if tok in self._common_replaces_refine]


class WordLevelPerturbatorRefined(WordLevelPerturbatorBase):
    def __init__(self):
        super().__init__()
        self._word_forms = self._refine_verbs()

    def _refine_verbs(self):
        verbs_refine = defaultdict(list)
        for src in self._word_forms:
            for tgt in self._word_forms[src]:
                edit_dist = editdistance.eval(tgt, src)
                if edit_dist > 2:
                    continue
                longer = max(len(src), len(tgt))
                if edit_dist / longer >= 0.5:
                    continue
                verbs_refine[src].append(tgt)

        return verbs_refine

    def _filter_tokens_to_delete(self, tokens: List[str]) -> List[int]:
        return [
            i
            for i, tok in enumerate(tokens)
            if tok in self._common_inserts and i > 0 and tokens[i - 1].lower() == tokens[i].lower()
        ]

    def _filter_tokens_to_replace(self, tokens: List[str]) -> List[int]:
        return [
            i
            for i, tok in enumerate(tokens)
            if tok in self._common_replaces_refine and tok.lower() not in {"not", "n't"}
        ]


class CharLevelPerturbator:
    def __init__(self):
        self._cache = {}
        self._n_types = 5
        self._max_iterations = 500

        with open(os.path.join(ROOT_PATH, "data/common_typo.json")) as file:
            self.common_typo = json.load(file)

    def __tokenize(self, sentence: str) -> Tuple[List[str], List[int]]:
        tokens = []
        word_idxs = []
        for idx, match in enumerate(re.finditer(r"([a-zA-Z]+)|([0-9]+)|.", sentence)):
            tok = match.group(0)
            tokens.append(tok)
            if len(tok) > 2 and tok.isalpha() and tok[0].islower():
                word_idxs.append(idx)
        return tokens, word_idxs

    def __detokenize(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def __sample_perturbations(self, word: str, n_samples: int) -> Set[str]:
        type_list = list(range(4)) * (n_samples // 4) + list(
            np.random.choice(self._n_types, n_samples % self._n_types, replace=False)
        )
        type_count = Counter(type_list)
        perturbations = set()
        for type in type_count:
            if type not in self._cache[word]:
                continue
            if len(self._cache[word][type]) >= type_count[type]:
                perturbations.update(set(sample(self._cache[word][type], type_count[type])))
            else:
                perturbations.update(self._cache[word][type])

        return perturbations

    def __get_perturbations(self, word: str, n_samples: int) -> Set[str]:
        if word not in self._cache:
            self._cache[word] = {}
            if word[0].islower():
                for modification_type in range(4):
                    self._cache[word][modification_type] = get_all_edit_dist_one(word, 10 ** modification_type)
                if word in self.common_typo:
                    self._cache[word][4] = set(self.common_typo[word])
            elif word[0].isupper():
                if word in self.common_typo:
                    self._cache[word][4] = set(self.common_typo[word])

        perturbations = self.__sample_perturbations(word, n_samples)

        return perturbations

    def get_local_neighbors(self, sentence: str, max_n_samples: int = 500) -> Set[str]:
        words, word_idxs = self.__tokenize(sentence)
        n_samples = min(len(word_idxs) * 20, max_n_samples)
        sent_perturbations = set()

        if len(word_idxs) == 0:
            return sent_perturbations

        for _ in range(self._max_iterations):
            word_idx = sample(word_idxs, 1)[0]
            words_cp = words[:]
            word_perturbations = list(self.__get_perturbations(words_cp[word_idx], n_samples=1))
            if len(word_perturbations) > 0:
                words_cp[word_idx] = word_perturbations[0]
                sent_perturbed = self.__detokenize(words_cp)
                if sent_perturbed != sentence:
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
                if sent_perturbed != sentence:
                    sent_perturbations.add(sent_perturbed)

                words_cp[word_idx] = word[:-1]
                sent_perturbed = self.__detokenize(words_cp)
                if sent_perturbed != sentence:
                    sent_perturbations.add(sent_perturbed)

        if len(sent_perturbations) > max_n_samples:
            sent_perturbations = list(sent_perturbations)
            np.random.shuffle(sent_perturbations)
            sent_perturbations = set(sent_perturbations[:max_n_samples])

        return sent_perturbations
