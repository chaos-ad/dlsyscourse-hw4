"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class EWiseSub(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a - b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, negate(out_grad)

def sub(a, b):
    return EWiseSub()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # assert self.scalar not in (0, 1)
        return out_grad * mul_scalar(power_scalar(node.inputs[0], self.scalar-1), self.scalar)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = out_grad * power_scalar(rhs, -1)
        rhs_grad = out_grad * mul_scalar(multiply(lhs, power_scalar(rhs, -2)), -1)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION`


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dims = len(a.shape)
        (swap1, swap2) = self.axes if self.axes is not None else (dims-1, dims-2)
        new_axes = list(range(dims))
        new_axes[swap1], new_axes[swap2] = new_axes[swap2], new_axes[swap1]
        return array_api.permute(a, new_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, node.op.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        out_shape = out_grad.shape
        padding = [1] * (len(out_shape) - len(in_shape))
        in_shape_padded = padding + list(in_shape)
        sum_along = []
        for index, (in_dim, out_dim) in enumerate(zip(in_shape_padded, out_shape)):
            if in_dim != out_dim:
                sum_along.append(index)
        sum_along = tuple(sum_along)
        return reshape(summation(out_grad, axes=sum_along), shape=in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

def get_padded_shape(original_shape, folded_axes_ids):
    if folded_axes_ids is None:
        folded_axes_ids = range(len(original_shape))
    elif isinstance(folded_axes_ids, tuple):
        folded_axes_ids = list(folded_axes_ids)
    elif isinstance(folded_axes_ids, int):
        folded_axes_ids = [folded_axes_ids]
    else:
        raise Exception(f"Unexpected argument type: {type(folded_axes_ids)=}")


    for idx, folded_axis_id in enumerate(folded_axes_ids):
        if folded_axis_id < 0:
            folded_axes_ids[idx] = len(original_shape) + folded_axis_id

    result = []
    for axis_idx, axis_size in enumerate(original_shape):
        if axis_idx in folded_axes_ids:
            result.append(1)
        else:
            result.append(axis_size)

    return result

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        result = a
        if isinstance(self.axes, (tuple, list)):
            for axis in reversed(sorted(self.axes)):
                result = array_api.summation(result, axis=axis)
        else:
            result = array_api.summation(result, axis=self.axes)
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        out_shape_padded = get_padded_shape(in_shape, self.axes)
        return broadcast_to(reshape(out_grad, shape=out_shape_padded), shape=in_shape)
        ### END YOUR SOLUTION

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        if len(lhs_grad.shape) > len(lhs.shape):
            axes = tuple(range(len(lhs_grad.shape) - len(lhs.shape)))
            lhs_grad = summation(lhs_grad, axes=axes)
        if len(rhs_grad.shape) > len(rhs.shape):
            axes = tuple(range(len(rhs_grad.shape) - len(rhs.shape)))
            rhs_grad = summation(rhs_grad, axes=axes)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * -1
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return mul_scalar(out_grad, -1)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * power_scalar(node.inputs[0], -1)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # ReLU is not twice-differentiable, so we will not need to use tensor ops here, just plain array_api
        output = node.realize_cached_data()
        grads = (output > 0)
        return out_grad * Tensor(grads, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
         # Keepdims makes broadcast work along right axes
        Z_max = array_api.max(Z, axis=self.axes)
        Z_max_reshaped = array_api.reshape(Z_max, get_padded_shape(Z.shape, self.axes))
        Z_max_broadcasted = array_api.broadcast_to(Z_max_reshaped, Z.shape)
        Z_sub = Z - Z_max_broadcasted
        result = array_api.log(array_api.summation(array_api.exp(Z_sub), axis=self.axes))
        result += Z_max
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].realize_cached_data()
        Z_max = array_api.max(Z, axis=self.axes)
        Z_max_reshaped = array_api.reshape(Z_max, get_padded_shape(Z.shape, self.axes))
        Z_max_broadcasted = array_api.broadcast_to(Z_max_reshaped, Z.shape)
        Z_sub = Tensor(Z - Z_max_broadcasted, device=node.inputs[0].device)
        Z_exp = exp(Z_sub)
        Z_norm = summation(Z_exp, axes=self.axes)
        Z_norm_reshaped = reshape(Z_norm, get_padded_shape(Z.shape, self.axes))
        Z_norm_broadcasted = broadcast_to(Z_norm_reshaped, Z.shape)
        Z_norm = Z_exp / Z_norm_broadcasted
        out_grad_reshaped = reshape(out_grad, get_padded_shape(Z.shape, self.axes))
        out_grad_broadcasted = broadcast_to(out_grad_reshaped, Z.shape)
        result = out_grad_broadcasted * Z_norm
        return result
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        y = exp(-x) + exp(x)
        z = mul_scalar(power_scalar(y, -2), 4)
        return out_grad * z
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        in_shape = args[0].shape
        out_shape = [len(args)] + list(in_shape)
        out = NDArray.make(out_shape, device=args[0].device)
        idxs = [slice(None, None, None) for j in range(len(in_shape))]
        for i, arg in enumerate(args):
            assert arg.shape == in_shape
            idxs_i = tuple([i] + idxs)
            out[idxs_i] = arg.compact()
        out_axes = list(range(1, len(out_shape)))
        out_axes.insert(self.axis, 0)
        return array_api.permute(out, tuple(out_axes)).compact()
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        in_shape = A.shape
        idx = [slice(None, None, None) for j in range(len(in_shape))]
        results = []
        for i in range(in_shape[self.axis]):
            idx_i = idx.copy()
            idx_i[self.axis] = i
            idx_i = tuple(idx_i)
            out = A[idx_i]
            # TODO: since it's a fake reduction, can I just drop the 1-dimension?
            out = array_api.summation(out, axis=self.axis)
            results.append(out)
        return tuple(results)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Tensor(out_grad.realize_cached_data().flip(self.axes), device=out_grad.device)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        view_idxs = []
        new_shape = []
        old_shape = a.shape
        for dim_idx, cur_dim_size in enumerate(old_shape):
            if dim_idx in self.axes:
                new_dim_size = cur_dim_size * (1 + self.dilation)
                new_shape.append(new_dim_size)
                view_idxs.append(slice(0, new_dim_size, 1 + self.dilation))
            else:
                new_shape.append(cur_dim_size)
                view_idxs.append(slice(0, cur_dim_size, 1))
        result = NDArray.make(new_shape, device=a.device)
        result.fill(0)
        result[tuple(view_idxs)] = a
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        view_idxs = []
        new_shape = []
        cur_shape = a.shape
        for dim_idx, cur_dim_size in enumerate(cur_shape):
            if dim_idx in self.axes:
                new_dim_size = cur_dim_size // (1 + self.dilation)
                new_shape.append(new_dim_size)
                view_idxs.append(slice(0, cur_dim_size, 1 + self.dilation))
            else:
                new_shape.append(cur_dim_size)
                view_idxs.append(slice(0, cur_dim_size, 1))
        result = NDArray.make(new_shape, device=a.device)
        res_idxs = [slice(None, None, None) for _ in range(len(new_shape))]
        result[tuple(res_idxs)] = a[tuple(view_idxs)]
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    ## Naive implementation:
    # N, H_in, W_in, C_in = A.shape
    # K, _, _, C_out = B.shape
    # assert B.shape[0] == B.shape[1], "Conv supports only square kernels"
    # assert B.shape[0] % 2 == 1, "Conv supports only odd-sized kernels"
    # assert B.shape[2] == A.shape[3], "Input channels are of mismatched size"

    # M = (K + 1) // 2 # margin of the kernel, the distance from center to the border
    # H_out, W_out = (H_in - (M-1) * 2), (W_in - (M-1) * 2)
    # output = NDArray.make(shape=(N, H_out, W_out, C_out), device=A.device)
    # output.fill(0)

    # for n in range(N):
    #     for c_in in range(C_in):
    #         for c_out in range(C_out):
    #             for h_out in range(H_out):
    #                 for w_out in range(W_out):

    #                     ## Calculating convolution for out[h_out, w_out]:
    #                     for h_kern in range(K):
    #                         for w_kern in range(K):
    #                             output[n,h_out,w_out,c_out] += A[n,h_out+h_kern,w_out+w_kern,c_in] * B[h_kern,w_kern,c_in,c_out]


    # return output

    def compute(self, X, W):
        ### BEGIN YOUR SOLUTION

        if self.padding > 0:
            padding = ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
            X = X.pad(padding)

        N, H_in, W_in, C_in = X.shape
        K, _, _, C_out = W.shape
        H_out, W_out = (H_in - K) // self.stride + 1, (W_in - K) // self.stride + 1
        
        # assert self.padding == 0 or W.shape[0] % 2 == 1, "Conv with padding supports only odd-sized kernels"
        assert W.shape[0] == W.shape[1], "Conv supports only square kernels"
        assert W.shape[2] == X.shape[3], "Input channels are of mismatched size"

        ## Calculating convolution with a single matmul via im2col trick:

        S = self.stride
        N_s, H_s, W_s, C_s = X.strides
        X_im2col = X.as_strided(shape=(N, H_out, W_out, K, K, C_in), strides=(N_s, (H_s * S), (W_s * S), H_s, W_s, C_s))
        X_im2col = X_im2col.compact().reshape((N * H_out * W_out, K * K * C_in)) ## compact uses extra O(K^2) memory
        output = X_im2col @ W.compact().reshape((K * K * C_in, C_out))
        output = output.reshape((N, H_out, W_out, C_out))
        return output
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs

        N, H_in, W_in, C_in = X.shape
        K, _, _, C_out = W.shape
        
        P = self.padding
        # print(f"{X.shape=}")
        # print(f"{W.shape=}")
        # print(f"{out_grad.shape=}")
        
        if self.stride > 1:
            out_grad_strided = dilate(out_grad, axes=(1,2), dilation=self.stride-1)
        else:
            out_grad_strided = out_grad
        # print(f"{out_grad_strided.shape=}")
        _, H_out, W_out, _ = out_grad_strided.shape

        W_flipped = flip(W, axes=(0,1))
        W_transposed = transpose(W_flipped, axes=(2,3))
        # print(f"{W_transposed.shape=}")
        X_grad_padding = K - P - 1
        # print(f"{X_grad_padding=}")
        X_grad = conv(out_grad_strided, W_transposed, stride=1, padding=X_grad_padding)
        # print(f"{X_grad.shape=}")
        
        X_perm = transpose(X, axes=(3,0))
        # print(f"{X_perm.shape=}")
        out_grad_perm = transpose(transpose(out_grad_strided, axes=(0,1)), axes=(1,2))
        # print(f"{out_grad_perm.shape=}")
        X_perm_padding = (H_out - H_in + K - 1) // 2
        # print(f"{X_perm_padding=}")
        W_grad_perm = conv(X_perm, out_grad_perm, stride=1, padding=X_perm_padding)
        # print(f"{W_grad_perm.shape=}")
        W_grad = transpose(transpose(W_grad_perm, axes=(0,1)), axes=(1,2))
        # print(f"{W_grad.shape=}")

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



