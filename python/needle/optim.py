"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param_id, param in enumerate(self.params):
            grad = ndl.Tensor(param.grad.detach(), device=param.grad.device, dtype=param.dtype) # Gradient
            grad += self.weight_decay * param.data.detach() # L2 Regularization term
            u_prev = self.u.get(param_id, ndl.init.zeros(*grad.shape, device=grad.device)) # Prev momentum
            u_curr = (self.momentum * u_prev) + (1 - self.momentum) * grad # Curr momentum
            self.u[param_id] = u_curr.detach()
            param.data -= self.lr * u_curr.detach()
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param_id, param in enumerate(self.params):
            grad = ndl.Tensor(param.grad.detach(), device=param.grad.device, dtype=param.dtype) # Gradient
            grad += self.weight_decay * param.data.detach() # L2 Regularization term
            grad_sq = ndl.ops.power_scalar(grad, 2)

            m_prev = self.m.get(param_id, ndl.init.zeros(*grad.shape, device=grad.device))
            m_curr = self.beta1 * m_prev + (1 - self.beta1) * grad     # running avg of grad

            v_prev = self.v.get(param_id, ndl.init.zeros(*grad.shape, device=grad.device))
            v_curr = self.beta2 * v_prev + (1 - self.beta2) * grad_sq  # running avg of grad^2

            self.m[param_id] = m_curr.detach()
            self.v[param_id] = v_curr.detach()

            m_curr_hat = m_curr.detach() / (1 - (self.beta1 ** self.t))
            v_curr_hat = v_curr.detach() / (1 - (self.beta2 ** self.t))

            param.data -= self.lr * m_curr_hat / (ndl.ops.power_scalar(v_curr_hat, 1/2) + self.eps)
        ### END YOUR SOLUTION
