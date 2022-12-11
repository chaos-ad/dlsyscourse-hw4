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

    a,b,k,s = 3,16,7,4
    device = ndl.cpu()
    np.random.seed(0)
    a = nn.Conv(a, b, k, stride=s, bias=True, device=device)
    b = nn.BatchNorm2d(dim=b, device=device)
    c = nn.ReLU()

    _A = np.random.randn(2, 3, 32, 32)
    A = ndl.Tensor(_A, device=device)
    y0 = A
    y1 = a(y0)
    y2 = b(y1)
    y3 = c(y2)
    y = y3

    assert np.linalg.norm(np.array([[-1.8912625 ,  0.64833605,  1.9400386 ,  1.1435282 ,  1.89777   ,
         2.9039745 , -0.10433993,  0.35458302, -0.5684191 ,  2.6178317 ],
       [-0.2905612 , -0.4147861 ,  0.90268034,  0.46530387,  1.3335679 ,
         1.8534894 , -0.1867125 , -2.4298222 , -0.5344223 ,  4.362149  ]]) - y.numpy()) < 1e-2

if __name__ == '__main__':
    main()