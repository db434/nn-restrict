# Word language model. Based on
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py

import torch.nn as nn

import modifiers.modules as quantisable
import structured.convert_to_conv as c2c
import util.log

# These models should work for any text dataset.
models = {"WikiText-2": ["wlm_lstm_large", "wlm_lstm_medium", "wlm_gru_large",
                         "wlm_gru_medium", "wlm_rnn_tanh_large",
                         "wlm_rnn_tanh_medium", "wlm_rnn_relu_large",
                         "wlm_rnn_relu_medium"]}


class WordLanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, num_tokens, num_inputs, num_hidden_units,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        """
        :param rnn_type: One of "LSTM", "GRU", "RNN_TANH", "RNN_RELU".
        :param num_tokens: Number of words in the dictionary.
        :param num_inputs: The number of features in each word embedding.
        :param num_hidden_units: Number of hidden units per layer.
        :param num_layers: Number of layers in the model.
        :param dropout: Dropout rate for all dropout layers.
        :param tie_weights: Use same weights for encoder and decoder.
        """
        super(WordLanguageModel, self).__init__()

        assert rnn_type in ["LSTM", "GRU", "RNN_TANH", "RNN_RELU"]

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_tokens, num_inputs)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(c2c, rnn_type)(num_inputs, num_hidden_units,
                                              num_layers, dropout=dropout)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            self.rnn = c2c.RNN(num_inputs, num_hidden_units, num_layers,
                               nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(num_hidden_units, num_tokens)

        # Quantisation is built into c2c modules, so just need to use it on the
        # Embedding's and decoder's outputs.
        self.quantise = quantisable.Quantiser()

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press &
        # Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for
        # Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            assert num_hidden_units == num_inputs
            # TODO: allow alternate implementations
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.num_hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.hidden = None

        self.init_weights()
        self.init_hidden(kwargs["args"].batch_size)

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        # TODO: allow alternate implementations
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            self.register_buffer("hidden0",
                                 weight.new_zeros(self.num_layers, batch_size,
                                                  self.num_hidden_units))
            self.register_buffer("hidden1",
                                 weight.new_zeros(self.num_layers, batch_size,
                                                  self.num_hidden_units))
            self.hidden = (self.hidden0, self.hidden1)
        else:
            self.register_buffer("hidden0",
                                 weight.new_zeros(self.num_layers, batch_size,
                                                  self.num_hidden_units))
            self.hidden = self.hidden0

    def repackage_hidden(self):
        """Wraps hidden states in new Tensors to detach them from their
        history."""
        if self.rnn_type == "LSTM":
            self.hidden0.detach()
            self.hidden1.detach()
            self.hidden = (self.hidden0, self.hidden1)
        else:
            self.hidden0.detach()
            self.hidden = self.hidden0

    def forward(self, x):
        # Ensure the size of the hidden state is compatible with the new input.
        _, in_batch_size = x.size()
        _, hidden_batch_size, _ = self.hidden0.size()
        if in_batch_size != hidden_batch_size:
            util.log.info("Changing batch size from", hidden_batch_size, "to",
                          in_batch_size)
            self.init_hidden(in_batch_size)

        # At the start of each batch, detach the hidden state from how it was
        # previously produced. If we didn't, the model would try
        # backpropagating all the way to start of the dataset.
        self.repackage_hidden()

        embedding = self.drop(self.quantise(self.encoder(x)))
        output, self.hidden = self.rnn(embedding, self.hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1),
                                           output.size(2)))

        return self.quantise(decoded)


# A few sensible defaults taken from the source. Use the above constructor
# directly if something specific is wanted.
def wlm_lstm_large(**kwargs):
    model = WordLanguageModel("LSTM", kwargs["args"].num_tokens, 1500, 1500, 2,
                              dropout=0.65, tie_weights=True, **kwargs)
    return model


def wlm_lstm_medium(**kwargs):
    model = WordLanguageModel("LSTM", kwargs["args"].num_tokens, 650, 650, 2,
                              dropout=0.5, tie_weights=True, **kwargs)
    return model


def wlm_gru_large(**kwargs):
    model = WordLanguageModel("GRU", kwargs["args"].num_tokens, 1500, 1500, 2,
                              dropout=0.65, tie_weights=True, **kwargs)
    return model


def wlm_gru_medium(**kwargs):
    model = WordLanguageModel("GRU", kwargs["args"].num_tokens, 650, 650, 2,
                              dropout=0.5, tie_weights=True, **kwargs)
    return model


def wlm_rnn_tanh_large(**kwargs):
    model = WordLanguageModel("RNN_TANH", kwargs["args"].num_tokens, 1500, 1500,
                              2, dropout=0.65, tie_weights=True, **kwargs)
    return model


def wlm_rnn_tanh_medium(**kwargs):
    model = WordLanguageModel("RNN_TANH", kwargs["args"].num_tokens, 650, 650,
                              2, dropout=0.5, tie_weights=True, **kwargs)
    return model


def wlm_rnn_relu_large(**kwargs):
    model = WordLanguageModel("RNN_RELU", kwargs["args"].num_tokens, 1500, 1500,
                              2, dropout=0.65, tie_weights=True, **kwargs)
    return model


def wlm_rnn_relu_medium(**kwargs):
    model = WordLanguageModel("RNN_RELU", kwargs["args"].num_tokens, 650, 650,
                              2, dropout=0.5, tie_weights=True, **kwargs)
    return model
