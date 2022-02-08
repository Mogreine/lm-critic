import re

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import sent_tokenize
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS, HYPHENS
from spacy.util import compile_infix_regex
from spacy.lang.en import English


class TextPreprocessor:
    def __init__(self):
        self.nlp = English()
        self.gec_tokenizer = self.__get_tokenizer()
        self.bea19_tokenizer = self.__get_tokenizer(True)

    def __get_tokenizer(self, is_bea19: bool = False):
        infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
        )
        if is_bea19:
            infixes += [r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS)]

        infix_re = compile_infix_regex(infixes)

        return Tokenizer(
            self.nlp.vocab,
            prefix_search=self.nlp.tokenizer.prefix_search,
            suffix_search=self.nlp.tokenizer.suffix_search,
            infix_finditer=infix_re.finditer,
            token_match=self.nlp.tokenizer.token_match,
            rules=self.nlp.Defaults.tokenizer_exceptions,
        )

    def preprocess(self, text: str, is_bea19: bool = False):
        self.nlp.tokenizer = self.gec_tokenizer if not is_bea19 else self.bea19_tokenizer
        return [str(w) for w in self.nlp(text)]


class TextPostprocessor:
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def handle_double_quote(cls, sent):
        cur_str = ""
        exp_left = True
        ignore_space = False
        for char in sent:
            if char == '"':
                if exp_left:  # this is a left "
                    cur_str = cur_str.rstrip() + ' "'
                    exp_left = not exp_left
                    ignore_space = True
                else:  # this is a right "
                    cur_str = cur_str.rstrip() + '" '
                    exp_left = not exp_left
                    ignore_space = False
            else:
                if ignore_space:  # expecting right
                    if char == " ":
                        continue
                    else:
                        cur_str = cur_str + char
                        ignore_space = False
                else:
                    cur_str = cur_str + char
        cur_str = cur_str.strip()
        cur_str = re.sub(r"[ ]+", " ", cur_str)
        return cur_str

    @classmethod
    def postprocess_space(cls, sent):
        sent = re.sub(r"[ ]+\.", ".", sent)
        sent = re.sub(r"[ ]+,", ",", sent)
        sent = re.sub(r"[ ]+!", "!", sent)
        sent = re.sub(r"[ ]+\?", "?", sent)
        sent = re.sub(r"\([ ]+", "(", sent)
        sent = re.sub(r"[ ]+\)", ")", sent)
        sent = re.sub(r" \'s( |\.|,|!|\?)", r"'s\1", sent)
        sent = re.sub(r"n \'t( |\.|,|!|\?)", r"n't\1", sent)
        return sent

    @classmethod
    def detokenize_sent(cls, sent):
        # Clean raw sent
        sent = re.sub(r"\' s ", "'s ", sent)
        toks = sent.split()
        if len([1 for t in toks if t == "'"]) % 2 == 0:
            toks = ['"' if t == "'" else t for t in toks]
        sent = " ".join(toks)
        #
        sents = sent_tokenize(sent)
        final_sents = []
        for _sent in sents:
            _sent = cls.detokenizer.detokenize(_sent.split())
            res = cls.handle_double_quote(_sent)
            if res == -1:
                print("unbalanced double quote")
                print(_sent)
            else:
                _sent = res
            final_sents.append(_sent)
        sent = " ".join(final_sents)
        sent = cls.postprocess_space(sent)
        return sent
