import os
import re
import torch
import numpy as np

def _assert_suffix_match(suffix, path):
    assert re.search(r"\.{}$".format(suffix), path), "suffix mismatch"

def make_parent_dir(filepath):
    parent_path = os.path.dirname(filepath)
    if not os.path.isdir(parent_path):
        try:
            os.mkdir(parent_path)
        except FileNotFoundError:
            make_parent_dir(parent_path)
            os.mkdir(parent_path)
        print("[INFO] Make new directory: '{}'".format(parent_path))

def load_from_pth(model, path):
    return model.load_state_dict(torch.load(path))

def save_to_pth(data, path, model=True):
    _assert_suffix_match("pth", path)
    make_parent_dir(path)
    if model:
        if hasattr(data, "module"):
            data = data.module.state_dict()
        else:
            data = data.state_dict()
    torch.save(data, path)
    print("[INFO] Save as pth: '{}'".format(path))
