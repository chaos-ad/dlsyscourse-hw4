import os
import sys
import numpy as np

sys.path.append('./python')

import needle as ndl
from needle import backend_ndarray as nd
import test_nd_backend

def main():
    print(f"{os.getpid()=}")
    device = ndl.cpu()
    a = np.arange(1, 25).reshape(3, 2, 4)
    A = nd.NDArray(a, device=device)
    print(f"{A=}")

    B = A.flip((0,2,))
    print(f"{B=}")

if __name__ == '__main__':
    main()