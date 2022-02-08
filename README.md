# LM critic
Implementation of LM critic for sentence evaluation.

# Dependencies
To install all the dependencies run from the project root:
```
pip install -r requirements.txt
```

# Usage
The critic is defined in `src/critic.py`. In order to evaluate a sentence run:
```
PYTHONPATH=. python src/evaluate_sentence.py -s $SENTENCE$ --use_gpu --refined 
```
where
* `-s` -- the sentence to evaluate
* `--bs` -- batch size of lm critic for probability calculation
* `--refined` -- enables refine word-level perturbations (preferable method)
* `--use_gpu` -- enables gpu usage for probabilities computation 

# Evaluation
To run evaluation on the dataset:
```
PYTHONPATH=. python src/eval_critic.py --seed $SEED$ --use_gpu --refined 
```
where
* `--seed` -- initialization state of a pseudo-random number generator
* `--bs` -- batch size of lm critic for probability calculation
* `--dataset` -- dataset ot evaluate on. Must be `bea19` or `realec`
* `--refined` -- enables refine word-level perturbations
* `--use_gpu` -- enables gpu usage for probabilities computation

## BEA19
You can find the dataset at `data/eval_data.jsonl`. It is the same dataset which is used 
for lm critic evaluation in the paper (contains sentences from GMEG-wiki, GMEG-yahoo and BEA19). 
### Recognize "Good"
| Method   |      P      |  R  | F_{0.5}|
|----------|:-------------:|:------:|:---:|
| ED1 + word(all)           | 67.8 | 16.6 | 41.9 |
| ED1 + word(all). Paper    | 69.7 | 10.2 | 32.2 |
| ED1 + word(refine)        | 68.2 | 76.1 | 69.6 |
| ED1 + word(refine). Paper | 68.4 | 75.5 | 69.7 |

### Recognize "Bad"
| Method   |      P      |  R  | F_{0.5}|
|----------|:-------------:|:------:|:---:|
| ED1 + word(all)           | 52.5 | 92.4 | 57.4 |
| ED1 + word(all). Paper    | 51.5 | 95.5 | 56.7 |
| ED1 + word(refine)        | 73.0 | 64.5 | 71.1 |
| ED1 + word(refine). Paper | 72.7 | 65.1 | 71.1 |

The results are comparable to the ones from the paper.

## REALEC style
You can find the dataset at `data/realec_style_eval.csv`.
REALEC style consists of 2000 pairs of good and bad sentences with only style mistakes.
### Recognize "Good"
| Method   |      P      |  R  | F_{0.5}|
|----------|:-------------:|:------:|:---:|
| ED1 + word(all)           | 58.7 | 10.6 | 30.9 |
| ED1 + word(refine)        | 54.8 | 48.5 | 53.4 |

### Recognize "Bad"
| Method   |      P      |  R  | F_{0.5}|
|----------|:-------------:|:------:|:---:|
| ED1 + word(all)           | 50.9 | 92.5 | 55.9 |
| ED1 + word(refine)        | 53.8 | 60.1 | 55.0 |

As expected, LM critic is not much better than a coin toss on style mistakes. That is because all the perturbations
either change orthography or word form/tense which won't generate a better sentence stylistically. 

Also style mistakes often cover a few words -- and there are no such perturbations for the critic.

# Drawbacks of critic
1. Character level perturbations quite often make non-existent words. It might be good to check if the word exists before creating such a perturbation.
2. There are word level perturbation like `I like apple.` -> `to I like apple.` With some insertions it is easy to understand if a perturbation is correct or not.
3. In general more rules for perturbation filtration (like refine or the ones described earlier) might improve the quality of lm critic.
4. Maybe there is a way to implement multi-word perturbation to cover style mistakes.