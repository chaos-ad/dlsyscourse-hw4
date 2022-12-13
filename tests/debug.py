import os
import sys
import logging
import numpy as np

sys.path.append('./python')

logger = logging.getLogger(__name__)

import needle as ndl
from needle import backend_ndarray as nd
import needle.nn as nn

def main():
    print(f"{os.getpid()=}")

    device = ndl.cpu()
   
    import test_sequence_models
    test_sequence_models.test_language_model_training(device=device)

if __name__ == '__main__':
    main()