"""
EigenWave-ASR: Compact Speech Recognition with Multi-Scale Robin Features
=========================================================================
27.8M parameter ASR | Learnable multi-scale temporal features | CTC + KenLM
"""

__version__ = "0.1.0"
__author__ = "Sakib Hasan"

from .model import EnhancedHybridASR, MultiScaleRobinFeatures
from .vocab import Vocab
