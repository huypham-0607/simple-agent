import torch

from utils import get_device
from torch import nn

device = get_device()

class LinearRegression(nn.Module):
    def __init__(self, num_features:int, out_features:int) -> None:
        super().__init__()
        self.lin = nn.Linear(
            in_features = num_features,
            out_features = 1,
            bias = True,
            device = device
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.lin(X)
        return output.squeeze(-1)

class RNN(nn.Module):
    def __init__(self, num_inputs: int, num_hiddens: int, sigma: float = 0.01):
        """Initialize RNN model - Hidden state computation loop.
        
        Args:
            - num_inputs: Dimension of a token vector
            - num_hiddens: Dimension of hidden states
            - sigma: Initialization hyperparameter for W_xh & W_hh
        
        Returns:
            - List of updated hidden states after each timestep
            - Final hidden state after performing updates for all time steps

        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma

        self.W_xh = nn.Parameter(
            torch.randn(self.num_inputs, self.num_hiddens) * sigma
        )
        self.W_hh = nn.Parameter(
            torch.randn(self.num_hiddens, self.num_hiddens) * sigma
        )
        self.b_h = nn.Parameter(
            torch.zeros(self.num_hiddens)
        )

    def forward(self, inputs: torch.Tensor, state = None) -> tuple:
        """Perform a RNN forward pass for given input, updating H_t at each step.
        
        Args:
            - inputs: tensor of [num_steps, batch_size, num_inputs], where
                - num_steps: No of steps we are processing (10 steps means 100 states update)
                - batch_size: No of sequence we are processing (5 means processing 5 sequences,
                  each with num_steps timesteps)
                - num_inputs: Vector dimension for each token.
            - state: Last state for our hidden state. Initialize to torch.zeros() if N/A
        
        Returns:
            - List of updated hidden states after each timestep
            - Final hidden state after performing updates for all time steps

        """
        if (state == None):
            state = torch.zeros(inputs.shape[1], self.num_hiddens, device=inputs.device)

        outputs = []
        for X in inputs:
            state = torch.tanh(X @ self.W_xh + state @ self.W_hh + self.b_h)
            outputs.append(state)
        return outputs, state


class RNNLM(nn.Module):
    """RRN-based language model"""

    def __init__(self, rnn, vocab_size, lr=0.01) -> None:
        """Intialize the RNN-based language model
        
        Args:
            - rnn: The RNN class for hidden state loop defined above
            - vocab_size: size of the corpus vocabulary
            - lr: learning rate
        """
        super().__init__()
        self.rnn = rnn
        self.vocab_size = vocab_size
        self.lr = lr
        self.init_params()


    def init_params(self) -> None:
        """Initialize parameters for the RNNLM"""
        self.W_hq = nn.Parameter(
            torch.randn(self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
        )
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))

    def one_hot(self, X):
        """Output a one-hot encoding of an input X

        Args:
            - X: An input tensor of shape (batch_size, num_steps)

        Output:
            - A tensor of shape (num_steps, batch_size, vocab_size)
        
        """
        return nn.functional.one_hot(X.T, self.vocab_size).type(torch.float32)
        
    def output_layer(self, rnn_outputs):
        """A fully connected layer to transform hidden state into token pred

        Args:
            - rnn_outputs: List of hidden state (batch_size, num_hidden) outputted.

        Returns:
            - List of predictions for each hidden state.
        """

        outputs = [hidden @ self.W_hq + self.b_q for hidden in rnn_outputs]
        return torch.stack(outputs,1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def predict(self, prefix, num_preds, vocab, device=device):
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor([[outputs[-1]]], device=device)
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1:
                outputs.append(vocab[prefix[i + 1]])
            else:
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(axis=2).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])