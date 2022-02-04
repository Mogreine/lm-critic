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
        infixes = [
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
        ]
        if is_bea19:
            infixes.append([r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS)])

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
