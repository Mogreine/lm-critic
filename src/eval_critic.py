import os
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser
from typing import Dict, List, Tuple
from sklearn.metrics import fbeta_score, precision_score, recall_score

from definitions import ROOT_PATH
from src.critic import LMCritic
from src.utils import seed_everything
from src.tokenizer import TextPostprocessor


def load_bea19(data_path: str) -> Tuple[List[str], List[str]]:
    good_sents, bad_sents = [], []
    for line in open(data_path):
        obj = json.loads(line)
        good_sents.append(obj["good"])
        bad_sents.append(obj["bad"])

    return good_sents, bad_sents


def load_realec(data_path: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(data_path)
    return df["good_sentence"].tolist(), df["bad_sentence"].tolist()


def calc_metrics(preds, target) -> Dict[str, float]:
    return {
        "precision": precision_score(target, preds),
        "recall": recall_score(target, preds),
        "f_0.5": fbeta_score(target, preds, beta=0.5),
    }


def evaluate(
    good_sentences, bad_sentences, batch_size: int = 64, use_gpu: bool = False, is_refined: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    critic = LMCritic(use_gpu=use_gpu)

    preds = []
    for sentence in tqdm(good_sentences + bad_sentences, desc="Evaluating sentences..."):
        sentence = TextPostprocessor.detokenize_sent(sentence)
        is_good, *_ = critic.evaluate_sentence(
            sentence, n_samples=100, batch_size=batch_size, return_counter_example=False, is_refined=is_refined
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
    args.add_argument("--bs", type=int, default=64, help="Batch size fo probability calculation")
    args.add_argument("--dataset", type=str, default="bea19", help="Dataset ot evaluate on. Must be bea19 or realec.")
    args.add_argument("--use_gpu", action="store_true")
    args.add_argument("--refined", action="store_true", help="Perturbation method")
    args = args.parse_args()

    assert args.dataset in ["bea19", "realec"], f"Unsupported dataset: {args.dataset}. Must be bea19 or realec."

    seed_everything(1)

    bea19_data_path = os.path.join(ROOT_PATH, "data/eval_data.jsonl")
    realec_data_path = os.path.join(ROOT_PATH, "data/realec_style_eval.csv")

    if args.dataset == "bea19":
        good_sentences, bad_sentences = load_bea19(bea19_data_path)
    else:
        good_sentences, bad_sentences = load_realec(realec_data_path)

    metrics_good, metrics_bad = evaluate(good_sentences, bad_sentences, args.bs, args.use_gpu, args.refined)

    print("Recognize good:")
    for name, value in metrics_good.items():
        print(f"{name}: {value: .3f}")

    print("\nRecognize bad:")
    for name, value in metrics_bad.items():
        print(f"{name}: {value: .3f}")
