from . import butterfly
from . import butterfly_old
from . import deep_roots
from . import depthwise_butterfly
from . import depthwise_separable
from . import depthwise_shuffle
from . import fully_connected
from . import hadamard
from . import shift
from . import shuffle

__all__ = ["butterfly", "deep_roots", "depthwise_separable", "fully_connected",
           "hadamard", "shift", "shuffle",
           "depthwise_butterfly", "depthwise_shuffle",
           "butterfly_old"]

conv2d_types = {
    'butterfly': butterfly.Conv2d,
    'butterfly_old': butterfly_old.Conv2d,
    'fc': fully_connected.Conv2d,
    'hadamard': hadamard.Conv2d,
    'roots': deep_roots.Conv2d,
    'separable': depthwise_separable.Conv2d,
    'separable_butterfly': depthwise_butterfly.Conv2d,
    'separable_shuffle': depthwise_shuffle.Conv2d,
    'shift': shift.Conv2d,
    'shuffle': shuffle.Conv2d,
}
