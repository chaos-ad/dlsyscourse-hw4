import os
import sys
import numpy as np

sys.path.append('./python')

import needle as ndl
from needle import backend_ndarray as nd
import test_conv



def main():
    print(f"{os.getpid()=}")
    test_conv.test_dilate_backward(params=test_conv.dilate_backward_params[0], device=ndl.cpu())

if __name__ == '__main__':
    main()