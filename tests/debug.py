import os
import sys
import logging
import numpy as np

sys.path.append('./python')

logger = logging.getLogger(__name__)

import needle as ndl
from needle import backend_ndarray as nd

def main():
    print(f"{os.getpid()=}")

    import test_conv
    test_conv.test_nn_conv_backward(*test_conv.conv_back_params[0], device=ndl.cpu())

if __name__ == '__main__':
    main()