import torch

from math import ceil
from typing import List
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.perturbations import WordLevelPerturbator, CharLevelPerturbator, WordLevelPerturbatorRefined
from src.tokenizer import TextPreprocessor
from src.utils import seed_everything


class LMCritic:
    def __init__(self, critic_model: str = "gpt2", use_gpu: bool = False):
        self._model = GPT2LMHeadModel.from_pretrained(critic_model)
        self.device = "cuda" if use_gpu else "cpu"
        self._model.to(self.device)
        self._model.eval()

        self._tokenizer = GPT2Tokenizer.from_pretrained(critic_model)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        self._preprocessor = TextPreprocessor()

        self._word_level_perturbator_all = WordLevelPerturbator()
        self._word_level_perturbator_refined = WordLevelPerturbatorRefined()

        self._char_level_perturbator = CharLevelPerturbator()

    def __calc_loss(self, logits: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction="none")
        bs, seq_len = labels.size()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(bs, seq_len - 1)
        loss = (loss * shift_mask).sum(dim=1)  # [bsize, ]

        return -loss

    @torch.inference_mode()
    def __get_probability(self, sentences: List[str], batch_size: int = 64) -> torch.Tensor:
        log_probs = []
        n_batches = ceil(len(sentences) / batch_size)
        for idx in range(n_batches):
            batch = [self._tokenizer.bos_token + s for s in sentences[idx * batch_size: (idx + 1) * batch_size]]
            sentences_encoded = self._tokenizer(batch, padding=True, return_tensors="pt")
            sentences_encoded = {k: v.to(self.device) for k, v in sentences_encoded.items()}

            output = self._model(**sentences_encoded, labels=sentences_encoded["input_ids"])
            logp = self.__calc_loss(output.logits, sentences_encoded["attention_mask"], sentences_encoded["input_ids"])
            log_probs.append(logp)

        log_probs = torch.hstack(log_probs)

        return log_probs

    def evaluate_sentence(
        self,
        sentence: str,
        preprocess_method: str = "gec",
        n_samples: int = 500,
        batch_size: int = 64,
        is_refined: bool = True,
        verbose: bool = False,
        return_counter_example: bool = False,
    ):
        """
        Evaluates whether the sentence's is grammatically correct or not.

        :param sentence: The sentence to evaluate.
        :param preprocess_method: Must be "bea19" if used specifically on "bea19" sentences, "gec" otherwise.
        :param n_samples: Number of perturbations to be made.
        :param batch_size: Batch size of lm critic for probability calculation.
        :param is_refined: Whether or not to use refine version of perturbations.
        :param verbose: Whether or not print process information.
        :param return_counter_example: Whether or not return counter examples if the sentences evaluated to be incorrect.

        :return: Tuple of critic's judgment as True of False and critic's score. In addition may return counter example --
                 a better sentence and its score.
        """
        assert preprocess_method == "gec" or preprocess_method == "bea19", "Unknown preprocessing method: {}"

        # seed_everything(47)

        sentence_tokenized = self._preprocessor.preprocess(sentence, preprocess_method == "bea19")

        perturbator = self._word_level_perturbator_refined if is_refined else self._word_level_perturbator_all

        sent_perturbations_w, orig_sent = perturbator.get_local_neighbors(
            sentence_tokenized, max_n_samples=n_samples // 2
        )
        sent_perturbations_c = self._char_level_perturbator.get_local_neighbors(orig_sent, max_n_samples=n_samples // 2)

        if verbose:
            print("#sent_perturbations (char-level)", len(sent_perturbations_c))
            print("#sent_perturbations (word-level)", len(sent_perturbations_w))
        sents = [orig_sent] + list(sent_perturbations_c.union(sent_perturbations_w))
        logps = self.__get_probability(sents, batch_size)

        best_idx = logps.argmax().item()
        is_good = best_idx == 0

        if return_counter_example:
            counter_example = None
            if not is_good:
                counter_example = (sents[best_idx], float(logps[best_idx]))
            return is_good, logps[0].item(), counter_example

        return is_good, logps[0].item()
