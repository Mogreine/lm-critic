import torch

from typing import List

from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class LMCritic:
    def __init__(self, critic_model: str = "gpt2", use_gpu: bool = False):
        self.model = GPT2LMHeadModel.from_pretrained(critic_model)
        self.device = "cuda" if use_gpu else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = GPT2Tokenizer.from_pretrained(critic_model)

        # Don't know why
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __calc_loss(self, logits: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        bs, seq_len = labels.size()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(bs, seq_len - 1)
        loss = (loss * shift_mask).sum(dim=1)  # [bsize, ]

        return loss

    @torch.inference_mode()
    def __get_probability(self, sentences: List[str]) -> List[float]:
        sentences = [self.tokenizer.bos_token + " " + s for s in sentences]
        sentences_encoded = self.tokenizer(sentences, padding=True, return_tensors="pt")
        sentences_encoded = {k: v.to(self.device) for k, v in sentences_encoded.items()}
        output = self.model(**sentences_encoded, labels=sentences_encoded["input_ids"])
        loss = self.__calc_loss(output.logits, sentences_encoded["attention_mask"], sentences_encoded["input_ids"])

        return loss

    def evaluate_sentence(self, sentence: str) -> float:
        ...
