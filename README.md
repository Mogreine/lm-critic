# LM critic
Implementation of LM critic for sentence evaluation.

# 1. Dependencies
To install all the dependencies run from the project root:
```
pip install -r requrements.txt
```

# 2. Usage
The critic is defined in `src/critic.py`. In order to evaluate a sentence run:
```
PYTHONPATH=. python src/evaluate_sentence.py -s $SENTENCE$ --use_gpu --refined 
```
where
* `-s` -- the sentence to evaluate
* `--refined` -- enables refine word-level perturbations (preferable method)
* `--use_gpu` -- enables gpu usage for probabilities computation 

# 3. Evaluation
You can find the evaluation dataset at `data/eval_data.jsonl`. In order to run evaluation on the 
dataset run:
```
PYTHONPATH=. python src/eval_critic.py --seed $SEED$ --use_gpu --refined 
```
where
* `--seed` -- 
* `--refined` -- enables refine word-level perturbations
* `--use_gpu` -- enables gpu usage for probabilities computation

## Results
### Recognize "Good"
| Method   |      P      |  R  | F_{0.5}|
|----------|:-------------:|:------:|:---:|
| ED1 + word(all)           | 71.9 | 14.8 | 40.7 |
| ED1 + word(all). Paper    | 69.7 | 10.2 | 32.2 |
| ED1 + word(refine)        | 67.4 | 75.8 | 68.9 |
| ED1 + word(refine). Paper | 68.4 | 75.5 | 69.7 |

### Recognize "Bad"
| Method   |      P      |  R  | F_{0.5}|
|----------|:-------------:|:------:|:---:|
| ED1 + word(all)           | 52.5 | 94.2 | 57.6 |
| ED1 + word(all). Paper    | 51.5 | 95.5 | 56.7 |
| ED1 + word(refine)        | 72.3 | 63.3 | 70.3 |
| ED1 + word(refine). Paper | 72.7 | 65.1 | 71.1 |

The results are comparable to the ones from the paper.