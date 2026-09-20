# RNN Recreation

Goal of this exercise is to create a Recurrent Neural Network as a **character-level language model**, and train it on a corpus consisting H.G.Wells' The Time Machine

Entire RNN implementation will contains no D2L abstraction, or torch.nn.RNN. Data pipeline might contains D2L abstraction for time-save

## Model structure

First we define two matrices $X_t \in \mathbb{R}^{n \times d}$ as input & $H_t \in \mathbb{R}^{n \times h}$ as the hidden state at timestep $t$. We also define $W_{xh} \in \mathbb{R}^{d \times h}$, $W_{hh} \in \mathbb{R}^{h \times h}$, $W_{hq} \in \mathbb{R}^{h \times q}$, and $B_h \in \mathbb{R}^{1 \times h}$ as our tunable parameters. Then:

$$ H_{t} = \phi(X_t W_{xh} + H_{t-1}W_{hh} + b_h) $$

$$ O_{t} = H_{t} W_{hq} + b_{q} $$


