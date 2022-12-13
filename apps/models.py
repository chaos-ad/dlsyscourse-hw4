import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

def ConvBS(a,b,k,s, device):
    return nn.Sequential(
        nn.Conv(a, b, k, stride=s, bias=True, device=device),
        nn.BatchNorm2d(dim=b, device=device),
        nn.ReLU()
    )

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.conv_bn = nn.Sequential(
            ConvBS(3,16,7,4,device=device),
            ConvBS(16,32,3,2,device=device),
            nn.Residual(
                nn.Sequential(
                    ConvBS(32,32,3,1,device=device),
                    ConvBS(32,32,3,1,device=device),
                )
            ),
            ConvBS(32,64,3,2,device=device),
            ConvBS(64,128,3,2,device=device),
            nn.Residual(
                nn.Sequential(
                    ConvBS(128,128,3,1,device=device),
                    ConvBS(128,128,3,1,device=device),
                )
            ),
            nn.Flatten(),
            nn.Linear(128,128,device=device),
            nn.ReLU(),
            nn.Linear(128,10,device=device)
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.conv_bn(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.embeddings_layer = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        self.linear_layer = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch_size = x.shape
        x = self.embeddings_layer(x) # x.shape = (seq_len, batch_size, embedding_size)
        (out, h) = self.seq_model(x, h) # out.shape = (seq_len, batch_size, hidden_size)
        out = out.reshape(shape=(seq_len * batch_size, self.hidden_size))
        out = self.linear_layer(out)
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)