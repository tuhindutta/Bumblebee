import warnings
warnings.filterwarnings('ignore')

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

del warnings, torch

from . import (
    model,
    trainer
)