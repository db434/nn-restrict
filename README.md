# nn-restrict
Explore different restrictions to neural networks.

**Not yet prepared for public use. Use at your own risk!**

Use `{mnist, cifar10, imagenet, wlm}.py --help` to see usage information.

## Structural restrictions
Control how individual layers are structured, usually by changing how each input influences each output.

These restrictions apply to the convolution layers. Linear/fully connected layers have been mapped to convolution layers for a selection of networks.

Options:
 * [Butterfly](https://dawn.cs.stanford.edu/2019/06/13/butterfly/)
 * [Deep roots](https://arxiv.org/abs/1605.06489)
 * [Depthwise-separable](https://arxiv.org/abs/1610.02357)
  * Also depthwise-butterfly and depthwise-shuffle
 * [Shift](https://arxiv.org/abs/1711.08141)
 * [Shuffle](https://arxiv.org/abs/1707.01083)

## Numerical restrictions
Control which values can be used in computations. Restrictions can be applied separately to weights/activations/gradients, and to isolated parts of the network.

Options:
 * Set min/max
 * Add random noise
 * Set precision (all fixed-point values are a multiple of the precision)
 * Apply an arbitrary function (not exposed through the command line interface)
