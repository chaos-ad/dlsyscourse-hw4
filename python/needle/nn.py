"""The module.
"""
import math
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(ops.transpose(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype))) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        result = ops.matmul(X, self.weight)
        if self.bias:
            result += ops.broadcast_to(self.bias, result.shape)
        return result
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        new_shape = [X.shape[0], 1]
        for dim_size in list(X.shape)[1:]:
            new_shape[1] *= dim_size
        new_shape = tuple(new_shape)
        return ops.reshape(X, new_shape)
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(ops.add_scalar(ops.exp(-x), 1), scalar=-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        result = x
        for module in self.modules:
            result = module(result)
        return result
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = logits
        classes = logits.shape[-1]
        axes = tuple(list(range(1, len(Z.shape))))
        y_one_hot = init.one_hot(classes, y, device=logits.device)
        Zy = ops.summation(ops.multiply(Z, y_one_hot), axes=axes)
        res = ops.logsumexp(Z, axes=axes) - Zy
        res = ops.divide_scalar(ops.summation(res), res.shape[0])
        return res
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            X_mean = ops.summation(X, axes=(0,)) / X.shape[0]
            X_sub = X - ops.broadcast_to(ops.reshape(X_mean, (1, X.shape[1])), X.shape)
            X_sub_sq = ops.power_scalar(X_sub, 2)
            X_sub_sq_sum = ops.summation(X_sub_sq, axes=(0,))
            X_var = X_sub_sq_sum / X.shape[0]
            X_var_eps = X_var + self.eps
            X_sigma = ops.power_scalar(X_var_eps, 1/2)
            X_norm = X_sub / ops.broadcast_to(ops.reshape(X_sigma, (1, X.shape[1])), X.shape)
            result = X_norm * ops.broadcast_to(self.weight, X.shape)
            result += ops.broadcast_to(self.bias, result.shape)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * X_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * X_var.detach()
        else:
            X_mean = self.running_mean.detach()
            X_var = self.running_var.detach()
            X_sub = X - ops.broadcast_to(ops.reshape(X_mean, (1, X.shape[1])), X.shape)
            X_var_eps = X_var + self.eps
            X_sigma = ops.power_scalar(X_var_eps, 1/2)
            X_norm = X_sub / ops.broadcast_to(ops.reshape(X_sigma, (1, X.shape[1])), X.shape)
            result = X_norm * ops.broadcast_to(self.weight, X.shape)
            result += ops.broadcast_to(self.bias, result.shape)
        return result
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device))
        self.bias = Parameter(init.zeros(1, self.dim, device=device))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_mean = ops.summation(X, axes=(1,)) / X.shape[1]
        X_mean = ops.reshape(X_mean, (X.shape[0], 1))
        X_sub = X - ops.broadcast_to(X_mean, X.shape)
        X_sub_sq = ops.power_scalar(X_sub, 2)
        X_sub_sq_sum = ops.summation(X_sub_sq, axes=(1,))
        X_var = X_sub_sq_sum / X.shape[1]
        X_var_eps = X_var + self.eps
        X_sigma = ops.power_scalar(X_var_eps, 1/2)
        X_sigma = ops.reshape(X_sigma, (X.shape[0], 1))
        X_norm = X_sub / ops.broadcast_to(X_sigma, X.shape)
        result = X_norm * ops.broadcast_to(self.weight, X_norm.shape)
        if self.bias:
            result += ops.broadcast_to(self.bias, result.shape)
        return result
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            B = init.randb(*X.shape, p=(1-self.p), device=X.device)
            X_norm = X / (1 - self.p)
            result = X_norm * B
        else:
            result = X
        return result
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        K,I,O = kernel_size, in_channels, out_channels
        self.weight = Parameter(init.kaiming_uniform(fan_in=K*K*I, fan_out=K*K*O, shape=(K,K,I,O), device=device, dtype=dtype))

        bound = math.pow(K*K*I, -1/2)
        self.bias = Parameter(init.rand(O, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        # NCHW -> NHCW -> NHWC
        x = ops.transpose(ops.transpose(x, axes=(1,2)), axes=(2,3))
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.kernel_size // 2)
        out = ops.transpose(ops.transpose(out, axes=(2,3)), axes=(1,2))

        if self.bias:
            bias = ops.broadcast_to(ops.reshape(self.bias, shape=(1,self.out_channels,1,1)), out.shape)
            out += bias
        
        return out
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        if nonlinearity == "tanh":
            self.activation_fn = Tanh()
        elif nonlinearity == "relu":
            self.activation_fn = ReLU()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        init_arg = math.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-init_arg, high=init_arg, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-init_arg, high=init_arg, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(hidden_size, low=-init_arg, high=init_arg, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, low=-init_arg, high=init_arg, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        h_in = h or init.zeros(batch_size, self.hidden_size, device=self.W_hh.device, dtype=self.W_hh.dtype)
        
        a = (h_in @ self.W_hh)
        if self.bias_hh:
            a += ops.broadcast_to(ops.reshape(self.bias_hh, shape=(1,self.hidden_size)), shape=a.shape)
        
        b = (X @ self.W_ih)
        if self.bias_ih:
            b += ops.broadcast_to(ops.reshape(self.bias_ih, shape=(1,self.hidden_size)), shape=b.shape)
        
        c = a + b
        h_out = self.activation_fn(c)
        
        return h_out
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.dtype = dtype
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size if layer_idx == 0 else hidden_size, 
                hidden_size, 
                bias, 
                nonlinearity, 
                device, 
                dtype
            ) for layer_idx in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, H0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        
        batch_size = X.shape[1]
        H0 = H0 or init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        H0 = list(ops.split(H0, axis=0))

        H_t = H0
        output = []
        for t, X_t in enumerate(list(ops.split(X, axis=0))):
            H_next = []
            for layer_idx, h_t in enumerate(H_t):
                # In a multi-layer RNN, the input x_t(l) of the l-th layer (l >= 2) is the hidden state h_t^(l-1) of the previous layer.
                h_next = self.rnn_cells[layer_idx](X_t, h_t)
                X_t = h_next
                H_next.append(h_next)
            output.append(h_next)
            H_t = H_next

        H_t = ops.stack(tuple(H_t), axis=0)
        output = ops.stack(tuple(output), axis=0)
        return output, H_t
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.dtype = dtype
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid_fn = Sigmoid()
        self.tanh_fn = Tanh()

        init_arg = math.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size * 4, low=-init_arg, high=init_arg, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size * 4, low=-init_arg, high=init_arg, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(hidden_size * 4, low=-init_arg, high=init_arg, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size * 4, low=-init_arg, high=init_arg, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        
        batch_size = X.shape[0]
        h0, c0 = h if h else (None, None)
        h_in = h0 or init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        c_in = c0 or init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)

        tmp = (h_in @ self.W_hh) # shape=(batch_size, 4*hidden_size)
        if self.bias_hh:
            tmp += ops.broadcast_to(ops.reshape(self.bias_hh, shape=(1,self.hidden_size * 4)), shape=tmp.shape)
        
        tmp += (X @ self.W_ih) # shape=(batch_size, 4*hidden_size)
        if self.bias_ih:
            tmp += ops.broadcast_to(ops.reshape(self.bias_ih, shape=(1,self.hidden_size * 4)), shape=tmp.shape)
        
        tmp_parts = ops.split(ops.reshape(tmp, (batch_size, 4, self.hidden_size)), axis=1)
        tmp_i = self.sigmoid_fn(tmp_parts[0])
        tmp_f = self.sigmoid_fn(tmp_parts[1])
        tmp_g = self.tanh_fn(tmp_parts[2])
        tmp_o = self.sigmoid_fn(tmp_parts[3])

        c_out = c_in * tmp_f + tmp_i * tmp_g
        h_out = self.tanh_fn(c_out) * tmp_o
        
        return h_out, c_out

        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.dtype = dtype
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [
            LSTMCell(
                input_size if layer_idx == 0 else hidden_size, 
                hidden_size, 
                bias, 
                device, 
                dtype
            ) for layer_idx in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION

        batch_size = X.shape[1]
        (H0, C0) = h if h else (None, None)
        H0 = H0 or init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        C0 = C0 or init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        H0 = list(ops.split(H0, axis=0))
        C0 = list(ops.split(C0, axis=0))

        H_t = H0
        C_t = C0
        output = []
        for t, X_t in enumerate(list(ops.split(X, axis=0))):
            H_next = []
            C_next = []
            for layer_idx, (h_t, c_t) in enumerate(zip(H_t, C_t)):
                h_next, c_next = self.lstm_cells[layer_idx](X_t, (h_t, c_t))
                X_t = h_next
                H_next.append(h_next)
                C_next.append(c_next)
            output.append(h_next)
            H_t = H_next
            C_t = C_next

        H_n = ops.stack(tuple(H_t), axis=0)
        C_n = ops.stack(tuple(C_t), axis=0)
        output = ops.stack(tuple(output), axis=0)
        return output, (H_n, C_n)

        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        # NOTE: Why don't I use requires_grad=True for params everywhere?
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0, std=1, device=device, dtype=dtype)) 
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        X = init.one_hot(self.num_embeddings, x.realize_cached_data().flat, device=self.device, dtype=self.dtype)
        E = X @ self.weight
        E = E.reshape(shape=(x.shape[0], x.shape[1], self.embedding_dim))
        return E
        ### END YOUR SOLUTION
