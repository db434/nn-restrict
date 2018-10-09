"""
Wrappers to convert a range of different layer types to convolutions. This
allows the structured convolution to be applied to a wider range of networks.
"""

import math
import torch
import torch.nn as nn

from . import fully_connected as fc
import modifiers.modules as quantisable


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, conv=fc.Conv2d,
                 args=None):
        super(Linear, self).__init__()
        self.conv = conv(in_features, out_features, kernel_size=1, bias=bias,
                         args=args, batch_norm=False)

    def forward(self, x):
        # Linear layers receive an input of (batch size x in_features)
        # Conv layers expect their input to also have channel width and height.
        batch_size, in_features = x.size()
        x = x.view(batch_size, in_features, 1, 1)

        x = self.conv(x)

        # And perform the reverse transform for the output.
        return x.view(batch_size, -1)


class RNNBase(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super(RNNBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.bidirectional = bidirectional

        self.recurrent_layers = []

        # A couple of options that I don't support yet.
        assert not batch_first
        assert not bidirectional

    def forward(self, x, hidden):
        """
        Apply this module to a given input, using the provided hidden state.

        :param x: Input data. Shape (sequence length, batch size, input size).
        :param hidden: Hidden state. Shape depends on implementation.
        :return: (output, updated hidden)
          Output shape (sequence length, batch size, hidden size).
          Hidden shape is identical to the input.
        """
        x = self.unflatten_input(x)
        hidden = self.unflatten_hidden(hidden)
        total_output = []

        # x is a sequence of inputs.
        for item in x:
            new_hidden = []
            current_input = item

            for pos, layer in enumerate(self.recurrent_layers):
                output = layer(current_input, hidden[pos])

                # Never apply dropout to final layer
                if (self.dropout is not None) and (pos < self.num_layers - 1):
                    if type(output) is tuple:
                        output = tuple(self.dropout(t) for t in output)
                    else:
                        output = self.dropout(output)

                new_hidden.append(output)

                if type(output) is tuple:
                    current_input = output[0]  # Just want h from LSTM (not c)
                else:
                    current_input = output

            total_output.append(current_input)
            hidden = new_hidden

        # Return (output, hidden)
        #  * Output is the hidden state of the final layer for each item in
        #    the input sequence
        #  * Hidden is the combined hidden state of all layers at the end of
        #    the sequence
        total_output = self.flatten_output(total_output)
        hidden = self.flatten_hidden(hidden)
        return total_output, hidden

    @classmethod
    def unflatten_input(cls, x):
        """
        Convert a single tensor into something which allows iteration over
        elements of its sequence.
        """
        # The tensor's dimensions are (sequence length, ...), so iteration works
        # by default.
        return x

    @classmethod
    def unflatten_hidden(cls, hidden):
        """
        Convert a single tensor into something which allows iteration over
        the state for each layer of the network.
        """
        # The hidden state's shape is (layer, batch, hidden unit), so default
        # iteration works.
        return hidden

    @classmethod
    def flatten_output(cls, x):
        return torch.stack(x)

    @classmethod
    def flatten_hidden(cls, hidden):
        return torch.stack(hidden)

    def reset_parameters(self):
        """Default weight initialisation for RNN networks."""
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 nonlinearity="tanh", bias=True, batch_first=False,
                 dropout=0, bidirectional=False, conv=fc.Conv2d, args=None):
        super(RNN, self).__init__(input_size, hidden_size,
                                  num_layers=num_layers, bias=bias,
                                  batch_first=batch_first, dropout=dropout,
                                  bidirectional=bidirectional)

        for i in range(num_layers):
            name = "layer_" + str(i)
            layer = RNNCell(input_size, hidden_size, bias=bias,
                            nonlinearity=nonlinearity, conv=conv, args=args)
            self.add_module(name, layer)
            self.recurrent_layers.append(layer)

        self.reset_parameters()


class LSTM(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False,
                 conv=fc.Conv2d, args=None):
        super(LSTM, self).__init__(input_size, hidden_size,
                                   num_layers=num_layers, bias=bias,
                                   batch_first=batch_first, dropout=dropout,
                                   bidirectional=bidirectional)

        for i in range(num_layers):
            name = "layer_" + str(i)
            layer = LSTMCell(input_size, hidden_size, bias=bias,
                             conv=conv, args=args)
            self.add_module(name, layer)
            self.recurrent_layers.append(layer)

        self.reset_parameters()

    @classmethod
    def unflatten_hidden(cls, hidden):
        # Hidden state is now a tuple of two tensors, each with size
        #   (layers, batch size, hidden units)
        # Need to convert to a list of tuples of two tensors, each with size
        #   (batch size, hidden units)
        h, c = hidden
        return list(zip(h, c))

    @classmethod
    def flatten_hidden(cls, hidden):
        # Hidden state is now a list of tuples of two tensors, each with size
        #   (batch size, hidden units)
        # Need to convert to a single tuple of two tensors, each with size
        #   (layers, batch size, hidden units)
        hidden = list(zip(*hidden))
        h, c = hidden[0], hidden[1]
        return torch.stack(h), torch.stack(c)


class GRU(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False,
                 conv=fc.Conv2d, args=None):
        super(GRU, self).__init__(input_size, hidden_size,
                                  num_layers=num_layers, bias=bias,
                                  batch_first=batch_first, dropout=dropout,
                                  bidirectional=bidirectional)

        for i in range(num_layers):
            name = "layer_" + str(i)
            layer = GRUCell(input_size, hidden_size, bias=bias,
                            conv=conv, args=args)
            self.add_module(name, layer)
            self.recurrent_layers.append(layer)

        self.reset_parameters()


class RNNCellBase(nn.Module):
    def __init__(self, input_size, hidden_size, internal_size, bias=True,
                 conv=fc.Conv2d, args=None):
        super(RNNCellBase, self).__init__()

        # I don't believe that both sets of biases are necessary, but torch's
        # source does use both.
        # https://github.com/pytorch/pytorch/blob/72e171dc52540093c8ad4b6b539ce30ea200e6fd/torch/nn/modules/rnn.py#L679

        self.linear_x = Linear(input_size, internal_size, bias=bias, conv=conv,
                               args=args)
        self.linear_h = Linear(hidden_size, internal_size, bias=bias, conv=conv,
                               args=args)

        # Default: no quantisation. Change the behaviour using
        # modifiers.numbers.restrict_activations().
        self.quantise = quantisable.Quantiser()

    def forward(self, *data):
        raise NotImplementedError


class RNNCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True,
                 nonlinearity="tanh", conv=fc.Conv2d, args=None):
        super(RNNCell, self).__init__(input_size, hidden_size, hidden_size,
                                      bias=bias, conv=conv, args=args)

        # TODO Fall back on torch's RNNCell if possible?

        # With inputs x and h, aim to compute:
        #   h' = nonlinearity(weights_x@x + bias_x + weights_h@h + bias_h)
        #
        # Break this into:
        #   h' = nonlinearity(Linear(x) + Linear(h))

        self.nonlinearity = {"tanh": nn.Tanh, "relu": nn.ReLU}[nonlinearity]

    def forward(self, x, hidden):
        # The outputs from the Linear modules are already quantised.
        x = self.nonlinearity(self.linear_x(x) + self.linear_h(hidden))
        return self.quantise(x)


class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, conv=fc.Conv2d,
                 args=None):
        super(LSTMCell, self).__init__(input_size, hidden_size, 4*hidden_size,
                                       bias=bias, conv=conv, args=args)

        # With inputs x, h and c, aim to compute:
        #   i = sigmoid(weights_xi@x + bias_xi + weights_hi@h + bias_hi)
        #   f = sigmoid(weights_xf@x + bias_xf + weights_hf@h + bias_hf)
        #   g = tanh(weights_xg@x + bias_xg + weights_hg@h + bias_hg)
        #   o = sigmoid(weights_xo@x + bias_xo + weights_ho@h + bias_ho)
        #   c' = f*c + i*g
        #   h' = o * tanh(c')
        #
        # There's lots of scope for combining operations here:
        #   x2 = Linear(x) [quadruple width]
        #   h2 = Linear(h) [quadruple width]
        #   i = sigmoid(x2[first quarter] + h2[first quarter])
        #   f = sigmoid(x2[second quarter] + h2[second quarter])
        #   g = tanh(x2[third quarter] + h2[third quarter])
        #   o = sigmoid(x2[fourth quarter] + h2[fourth quarter])
        #   c' = f*c + i*g
        #   h' = o * tanh(c')

    def forward(self, x, state):
        h, c = state

        # These are already quantised.
        xi, xf, xg, xo = self.linear_x(x).chunk(4, 1)
        hi, hf, hg, ho = self.linear_h(h).chunk(4, 1)

        i = self.quantise(torch.sigmoid(xi + hi))
        f = self.quantise(torch.sigmoid(xf + hf))
        g = self.quantise(torch.tanh(xg + hg))
        o = self.quantise(torch.sigmoid(xo + ho))

        c2 = self.quantise(f*c + i*g)
        h2 = self.quantise(o * torch.tanh(c2))

        return h2, c2


class GRUCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, conv=fc.Conv2d,
                 args=None):
        super(GRUCell, self).__init__(input_size, hidden_size, 3*hidden_size,
                                      bias=bias, conv=conv, args=args)

        # With inputs x and h, aim to compute:
        #   r = sigmoid(weights_xr@x + bias_xr + weights_hr@h + bias_hr)
        #   z = sigmoid(weights_xz@x + bias_xz + weights_hz@h + bias_hz)
        #   n = tanh(weights_xn@x + bias_xn + r * (weights_hn@h + bias_hn))
        #   h' = (1 - z)*n + z*h
        #
        # There's lots of scope for combining operations here:
        #   x2 = Linear(x) [triple width]
        #   h2 = Linear(h) [triple width]
        #   r = sigmoid(x2[first third] + h2[first third])
        #   z = sigmoid(x2[second third] + h2[second third])
        #   n = tanh(x2[final third] + r * h2[final third])
        #   h' = (1 - z)*n + z*h

    def forward(self, x, hidden):
        # These are already quantised.
        xr, xz, xn = self.linear_x(x).chunk(3, 1)
        hr, hz, hn = self.linear_h(hidden).chunk(3, 1)

        r = self.quantise(torch.sigmoid(xr + hr))
        z = self.quantise(torch.sigmoid(xz + hz))
        n = self.quantise(torch.tanh(xn + r * hn))

        return (1 - z)*n + z*hidden
