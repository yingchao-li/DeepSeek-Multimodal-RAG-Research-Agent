"""Conservative native thread defaults for the macOS CPU runtime.

The original Apple Python environment can crash when torch/FAISS/BLAS all spawn
their default thread pools. Respect explicit user overrides.
"""
import os
import sys

if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
