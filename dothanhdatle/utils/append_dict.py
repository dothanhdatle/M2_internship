import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations


def append_elbo_dict(ld_old, ld_new):
    """
    Update loss dict
    """
    for k, v in ld_new.items():
        # Convert tensor to numpy array if necessary
        v = v.detach().cpu().numpy() if torch.is_tensor(v) else v
        # Append or create the list
        ld_old.setdefault(k, []).append(v)

    return ld_old
