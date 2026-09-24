"""Building blocks shared by several models."""
from keras import ops


def reflect_pad(x, pad=1):
    """Reflection-pad the two spatial dimensions of an NHWC tensor."""
    return ops.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="reflect")


def zero_pad(x, pad=1):
    """Zero-pad the two spatial dimensions of an NHWC tensor."""
    return ops.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])


def split_channels(x):
    """Split an NHWC tensor into two equally sized halves along the channel axis."""
    half = x.shape[-1] // 2
    return x[..., :half], x[..., half:2 * half]
