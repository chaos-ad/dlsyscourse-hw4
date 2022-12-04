import os
import sys
import numpy as np

sys.path.append('./python')

import needle as ndl
from needle import backend_ndarray as nd
import test_nd_backend

def main():
    print(f"{os.getpid()=}")

    # shape = (2, 4, 2)
    # device = ndl.cpu()
    # A = np.arange(nd.prod(shape)).reshape(*shape).astype(np.float32)
    # B = nd.array(A)

    # A = A[0, 1::2, :]
    # print(f"{A=}")
    # B = B[0, 1::2, :]
    # print(f"{B=}")

    SUMMATION_PARAMETERS = [
        ((1, 1, 1), None),
        ((5, 3), 0),
        ((8, 3, 2), 1),
        ((8, 3, 2), 2)
    ]
    for (shape, axes) in SUMMATION_PARAMETERS:
        test_nd_backend.test_summation_backward(shape, axes, device=ndl.cpu())

if __name__ == '__main__':
    main()