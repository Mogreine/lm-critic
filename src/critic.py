import torch

from typing import List

from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.tokenizer import TextPreprocessor


class LMCritic:
    def __init__(self, critic_model: str = "gpt2", use_gpu: bool = False):
        self.model = GPT2LMHeadModel.from_pretrained(critic_model)
        self.device = "cuda" if use_gpu else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = GPT2Tokenizer.from_pretrained(critic_model)
        # Don't know why
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.preprocessor = TextPreprocessor()

    def __calc_loss(self, logits: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        bs, seq_len = labels.size()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(bs, seq_len - 1)
        loss = (loss * shift_mask).sum(dim=1)  # [bsize, ]

        return loss

    @torch.inference_mode()
    def __get_probability(self, sentences: List[str]) -> torch.Tensor:
        sentences = [self.tokenizer.bos_token + " " + s for s in sentences]
        sentences_encoded = self.tokenizer(sentences, padding=True, return_tensors="pt")
        sentences_encoded = {k: v.to(self.device) for k, v in sentences_encoded.items()}
        output = self.model(**sentences_encoded, labels=sentences_encoded["input_ids"])
        loss = self.__calc_loss(output.logits, sentences_encoded["attention_mask"], sentences_encoded["input_ids"])

        return loss

    def evaluate_sentence(self, sentence: str, preprocess_method: str = "gec", verbose: bool = False, return_counter_example: bool = False) -> float:
        assert preprocess_method == "gec" or preprocess_method == "bea19", "Unknown preprocessing method: {}"

        sentence_tokenized = self.preprocessor.preprocess(sentence, preprocess_method == "bea19")

        sent_perturbations_w, orig_sent = get_local_neighbors_word_level(sent_toked, max_n_samples=n_samples // 2,
                                                                         mode=word_level_mode)
        sent_perturbations_c = get_local_neighbors_char_level(orig_sent, max_n_samples=n_samples // 2)
        if verbose > 1:
            print("#sent_perturbations (char-level)", len(sent_perturbations_c))
            print("#sent_perturbations (word-level)", len(sent_perturbations_w))
        sents = [orig_sent] + list(sent_perturbations_c.union(sent_perturbations_w))
        logps = self.__get_probability(sents)

        if logps is None:
            if verbose:
                print('Invalid input. Maybe the sentence is too long.')
            return None

        best_idx = logps.argmax().item()
        is_good = best_idx == 0

        if return_counter_example:
            counter_example = None
            if not is_good:
                counter_example = (sents[best_idx], float(logps[best_idx]))
            return is_good, logps[0].item(), counter_example

        return is_good, logps[0].item()
