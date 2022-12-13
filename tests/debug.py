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

    SEQ_LENGTHS = [1, 13]
    NUM_LAYERS = [1, 2]
    BATCH_SIZES = [1, 15]
    INPUT_SIZES = [1, 11]
    HIDDEN_SIZES = [1, 12]
    BIAS = [True, False]
    INIT_HIDDEN = [True, False]

    seq_length = SEQ_LENGTHS[0]
    num_layers = NUM_LAYERS[0]
    batch_size = BATCH_SIZES[0]
    input_size = INPUT_SIZES[0]
    hidden_size = HIDDEN_SIZES[0]
    bias = BIAS[0]
    init_hidden = INIT_HIDDEN[1]
    device = ndl.cpu()

    import test_sequence_models
    test_sequence_models.test_lstm(seq_length, num_layers, batch_size, input_size, hidden_size, bias, init_hidden, device)

if __name__ == '__main__':
    main()