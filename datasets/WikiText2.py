import torch
import torchtext
import torchtext.data as data

import locations


# TODO subclass an abstract Dataset class.
# Perhaps also a TextDataset class.
class WikiText2(object):

    # Some sensible defaults.
    name = "WikiText-2"
    default_model = "wlm_lstm_medium"
    location = None

    # These defaults only apply to models doing one particular task. If the
    # dataset is used in a different way, these may not be appropriate.
    default_lr = 20
    default_lr_steps = [(10, 0.25), (5, 0.25), (5, 0.25)]
    default_epochs = 25

    default_sequence_length = 35

    # Preprocessed state.
    # _text describes how to interpret the text in the dataset.
    # _train, _val and _test hold different pieces of the dataset.
    _text = None
    _train = None
    _val = None
    _test = None

    @staticmethod
    def num_tokens():
        WikiText2._init()
        return len(WikiText2._text.vocab)

    @staticmethod
    def word_to_token(word):
        """Convert a string to an identifying integer."""
        WikiText2._init()
        return WikiText2._text.vocab.stoi[word]

    @staticmethod
    def token_to_word(token):
        """
        Convert an identifying integer to a string.

        There are two special strings which may be encountered:
         * <eos> represents the end of stream
         * <unk> represents an unknown word
        """
        WikiText2._init()
        return WikiText2._text.vocab.itos[token]

    # Input channels and classes don't mean very much for text, but the
    # analogy for both of them is the number of words in the dictionary.
    @staticmethod
    def input_channels():
        return WikiText2.num_tokens()

    @staticmethod
    def num_classes():
        return WikiText2.num_tokens()

    @staticmethod
    def data_loaders(num_workers, batch_size, distributed=False):
        """Return train and validation data loaders for the WMT dataset."""
        return WikiText2.train_loader(num_workers, batch_size, distributed), \
            WikiText2.val_loader(num_workers, batch_size)

    @staticmethod
    def train_loader(num_workers, batch_size, distributed):
        # No support for distributed training yet.
        assert not distributed

        WikiText2._init()

        # Some weird notation because we have tuples of length 1.
        iterator, = data.BPTTIterator.splits(
            (WikiText2._train,), batch_size=batch_size, shuffle=True,
            bptt_len=WikiText2.default_sequence_length,
            sort_key=lambda x: len(x.text))
        return IteratorAdapter(iterator, num_workers=num_workers)

    @staticmethod
    def val_loader(num_workers, batch_size):
        WikiText2._init()

        # Some weird notation because we have tuples of length 1.
        iterator, = data.BPTTIterator.splits(
            (WikiText2._val,), batch_size=batch_size,
            bptt_len=WikiText2.default_sequence_length,
            sort_key=lambda x: len(x.text))
        return IteratorAdapter(iterator, num_workers=num_workers)

    @staticmethod
    def test_loader(num_workers, batch_size):
        WikiText2._init()

        # Some weird notation because we have tuples of length 1.
        iterator, = data.BPTTIterator.splits(
            (WikiText2._test,), batch_size=batch_size,
            bptt_len=WikiText2.default_sequence_length,
            sort_key=lambda x: len(x.text))
        return IteratorAdapter(iterator, num_workers=num_workers)

    @staticmethod
    def _init():
        if WikiText2._text is not None:
            return

        # Set up field: describe how text will be interpreted.
        WikiText2._text = data.Field(lower=True, batch_first=True)

        # Make splits for data.
        WikiText2._train, WikiText2._val, WikiText2._test = \
            torchtext.datasets.WikiText2.splits(WikiText2._text)

        # Build the vocabulary.
        WikiText2._text.build_vocab(WikiText2._train)


class IteratorAdapter(torch.utils.data.DataLoader):
    """
    Class which wraps torchtext's Iterator to create a DataLoader.
    """

    def __init__(self, iterator, num_workers):
        # TODO: pass more information to the superclass?
        # The iterator already handles shuffling and batches.
        super(IteratorAdapter, self).__init__(iterator.dataset,
                                              num_workers=num_workers,
                                              pin_memory=True)
        self.iterator = iterator

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        for batch in iter(self.iterator):
            yield (batch.text, batch.target.flatten())
