# LM critic
Implementation of LM critic for sentence evaluation.

# 1. Dependencies
To install all the dependencies run from the project root:
```
pip install -r requirements.txt
```

# 2. Usage
The critic is defined in `src/critic.py`. In order to evaluate a sentence run:
```
PYTHONPATH=. python src/evaluate_sentence.py -s $SENTENCE$ --use_gpu --refined 
```
where
* `-s` -- the sentence to evaluate
* `--bs` -- batch size of lm critic for probability calculation
* `--refined` -- enables refine word-level perturbations (preferable method)
* `--use_gpu` -- enables gpu usage for probabilities computation 

# 3. Evaluation
You can find the evaluation dataset at `data/eval_data.jsonl`. It is the same dataset which is used 
for lm critic evaluation in the paper (contains sentences from GMEG-wiki, GMEG-yahoo and BEA19). 
To run evaluation on the dataset:
```
PYTHONPATH=. python src/eval_critic.py --seed $SEED$ --use_gpu --refined 
```
where
* `--seed` -- initialization state of a pseudo-random number generator
* `--bs` -- batch size of lm critic for probability calculation
* `--refined` -- enables refine word-level perturbations
* `--use_gpu` -- enables gpu usage for probabilities computation

## Results
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