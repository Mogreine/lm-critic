import os
import string
import random
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_all_edit_dist_one(word, filetype=1111, sub_restrict=None):
    """
    Allowable edit_dist_one perturbations:
        1. Insert any lowercase characer at any position other than the start
        2. Delete any character other than the first one
        3. Substitute any lowercase character for any other lowercase letter other than the start
        4. Swap adjacent characters
    We also include the original word. Filetype determines which of the allowable perturbations to use.
    """
    insert, delete, substitute, swap = process_filetype(filetype)
    # last_mod_pos is last thing you could insert before
    last_mod_pos = len(word)  # - 1
    ed1 = set()
    if len(word) <= 2 or word[:1].isupper() or word[:1].isnumeric():
        return ed1
    for pos in range(1, last_mod_pos + 1):  # can add letters at the end
        if delete and pos < last_mod_pos:
            deletion = word[:pos] + word[pos + 1 :]
            ed1.add(deletion)
        if swap and pos < last_mod_pos - 1:
            # swapping thing at pos with thing at pos + 1
            swaped = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
            ed1.add(swaped)
        for letter in string.ascii_lowercase:  # +"'-": #no need to add '-, as we want to corrupt good to bad
            if insert:
                # Insert right after pos - 1
                insertion = word[:pos] + letter + word[pos:]
                ed1.add(insertion)
            can_substitute = sub_restrict is None or letter in sub_restrict[word[pos]]
            if substitute and pos < last_mod_pos and can_substitute:
                substitution = word[:pos] + letter + word[pos + 1 :]
                ed1.add(substitution)
    # Include original word
    # ed1.add(word)
    return ed1


def process_filetype(filetype):
    insert = (filetype // 1000) % 2 == 1
    delete = (filetype // 100) % 2 == 1
    substitute = (filetype // 10) % 2 == 1
    swap = filetype % 2 == 1
    return insert, delete, substitute, swap
