"""Model zoo. Use :func:`build_model` to construct any architecture by name."""
from . import NOAHTCV, Dynamic_PlainNet, Dynamic_UNet_simple, Megvii, MoDeNet, PlainNet, ResNet_18, SplitterNet

_BUILDERS = {
    "SplitterNet": lambda shape, f, enc, dec: SplitterNet.DYNUnet(shape, f),
    "SplitterNet_LN": lambda shape, f, enc, dec: SplitterNet.DYNUnet(shape, f, layer_norm=True),
    "MoDeNet": lambda shape, f, enc, dec: MoDeNet.DYNUnet(shape, enc, dec, 1, f),
    "Dynamic_PlainNet": lambda shape, f, enc, dec: Dynamic_PlainNet.DYNUnet(shape, enc, dec, 1, f),
    "Dynamic_UNet_simple": lambda shape, f, enc, dec: Dynamic_UNet_simple.DYNUnet(shape, enc, dec, 1, f),
    "Megvii": lambda shape, f, enc, dec: Megvii.DYNUnet(shape, enc, dec, f),
    "NOAHTCV": lambda shape, f, enc, dec: NOAHTCV.Unet(shape, f),
    "PlainNet": lambda shape, f, enc, dec: PlainNet.UNet(shape, f),
    "ResNet_18": lambda shape, f, enc, dec: ResNet_18.ResNet18_Denoiser(shape),
}

MODEL_NAMES = tuple(_BUILDERS)


def build_model(name, input_shape=(None, None, 3), num_filters=32, enc_blocks=(1, 1, 1, 1), dec_blocks=(1, 1, 1, 1)):
    """Build a denoising model by name.

    ``enc_blocks``/``dec_blocks`` (blocks per U-Net stage) are only used by the dynamic
    models (MoDeNet, Dynamic_PlainNet, Dynamic_UNet_simple, Megvii); ``num_filters`` is
    ignored by ResNet_18.
    """
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model {name!r}. Choose one of: {', '.join(MODEL_NAMES)}")
    return _BUILDERS[name](tuple(input_shape), int(num_filters), list(enc_blocks), list(dec_blocks))
