import json
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import torch
import numpy as np
import random
import pickle
import os

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_from_pickle(path):
    with open(path, "rb") as f:
        print(f'load the data from - {path}')
        return pickle.load(f)


def dump_to_pickle(path, obj, reset):
    if reset or not os.path.exists(path):
        with open(path, "wb") as f:
            print(f'save the data to - {path}')
            pickle.dump(obj, f)


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()