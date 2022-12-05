import os
import sys
import numpy as np

sys.path.append('./python')

import needle as ndl
from needle import backend_ndarray as nd
import test_nd_backend

def backward_check(f, *args, **kwargs):
    eps = 1e-3
    out = f(*args, **kwargs)
    c = np.arange(nd.prod(out.shape)).reshape(*out.shape)
    is_stacked = False
    if isinstance(args[0], list):
        args = args[0]
        is_stacked = True
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            if is_stacked:
                f1 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            if is_stacked:
                f2 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    if isinstance(backward_grad[0], ndl.TensorTuple): # TODO keep this?
        backward_grad = backward_grad[0].tuple()
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    for i in range(len(args)):
        print(f"{backward_grad[i].numpy().shape=}, {backward_grad[i].numpy()=}")
        print(f"{numerical_grad[i].shape=}, {numerical_grad[i]=}")
        print(f"{np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])=}")
    assert error < 1e-2
    return [g.numpy() for g in backward_grad]

def main():
    print(f"{os.getpid()=}")

    stack_back_params = [
        ((2, 2), 2, 0)
        # ( (3, 4), 3, 0),
        # ( (3, 4), 3, 1),
        # ( (3, 4), 3, 2),
        # ( (3, 4), 5, 2),
        # ( (3, 4), 1, 2),
    ]
    device = ndl.cpu()
    for shape, n, axis in stack_back_params:
        np.random.seed(0)
        get_tensor = lambda shape, i: ndl.Tensor((np.arange(nd.prod(shape)) + nd.prod(shape)*i).reshape(*shape), device=device)
        input_tensors = [get_tensor(shape, i) for i in range(n)]
        backward_check(ndl.stack, input_tensors, axis=axis)

if __name__ == '__main__':
    main()