import os
import json
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import fbeta_score, precision_score, recall_score

from definitions import ROOT_PATH
from src.critic import LMCritic
from src.other_utils import seed_everything
from src.tokenizer import TextPostprocessor


def load_data(data_path: str) -> Tuple[List[str], List[str]]:
    good_sents, bad_sents = [], []
    for line in open(data_path):
        obj = json.loads(line)
        good_sents.append(obj["good"])
        bad_sents.append(obj["bad"])

    return good_sents[:5], bad_sents[:5]


def calc_metrics(preds, target) -> Dict[str, float]:
    return {
        "precision": precision_score(target, preds),
        "recall": recall_score(target, preds),
        "f_0.5": fbeta_score(target, preds, beta=0.5),
    }


def evaluate(
    good_sentences, bad_sentences, use_gpu: bool = False, is_refined: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    critic = LMCritic(use_gpu=use_gpu)

    preds = []
    for sentence in tqdm(good_sentences + bad_sentences, desc="Evaluating sentences..."):
        sentence = TextPostprocessor.detokenize_sent(sentence)
        is_good, *_ = critic.evaluate_sentence(
            sentence, n_samples=100, return_counter_example=False, is_refined=is_refined
        )
        preds.append(is_good)

    target = [1] * len(good_sentences) + [0] * len(bad_sentences)

    metrics_good = calc_metrics(preds, target)

    # 0, 1 -> 1, 0
    preds = np.abs(np.array(preds) - 1)
    target = np.abs(np.array(target) - 1)
    metrics_bad = calc_metrics(preds, target)

    return metrics_good, metrics_bad


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--seed", type=int, default=1)
    args.add_argument("--use_gpu", action="store_true")
    args.add_argument("--refined", type=bool, default=True, help="Perturbation method")
    args = args.parse_args()

    seed_everything(1)

    data_path = os.path.join(ROOT_PATH, "artifacts/eval_data.jsonl")

    good_sentences, bad_sentences = load_data(data_path)

    metrics_good, metrics_bad = evaluate(good_sentences, bad_sentences, args.use_gpu, args.refined)

    print("Good:")
    for name, value in metrics_good.items():
        print(f"{name}: {value: .3f}")

    print("Bad:")
    for name, value in metrics_bad.items():
        print(f"{name}: {value: .3f}")
