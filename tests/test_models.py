import os

import keras
import numpy as np
import pytest

from models import MODEL_NAMES, build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "model_weights", "SplitterNet_MIDD_model.h5")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_build_and_preserve_shape(name):
    model = build_model(name, num_filters=16)
    x = np.random.rand(2, 64, 96, 3).astype("float32")
    assert model(x, training=False).shape == x.shape


@pytest.mark.parametrize("name", ["MoDeNet", "Dynamic_PlainNet", "Dynamic_UNet_simple", "Megvii"])
def test_dynamic_block_configuration(name):
    small = build_model(name, num_filters=16, enc_blocks=[1, 1, 1, 1], dec_blocks=[1, 1, 1, 1])
    large = build_model(name, num_filters=16, enc_blocks=[2, 1, 1, 2], dec_blocks=[1, 1, 1, 1])
    assert large.count_params() > small.count_params()


def test_pretrained_splitternet_matches_reference():
    """The pretrained MIDD weights reproduce the outputs of the original (TF 2.14) implementation."""
    ref = np.load(os.path.join(ROOT, "tests", "data", "splitternet_midd_reference.npz"))
    model = build_model("SplitterNet", num_filters=32)
    model.load_weights(WEIGHTS)
    assert model.count_params() == 731_059
    np.testing.assert_allclose(model(ref["x"], training=False), ref["y"], atol=1e-5)


@pytest.mark.parametrize("name", ["SplitterNet", "SplitterNet_LN", "MoDeNet"])
def test_keras_save_load_roundtrip(name, tmp_path):
    model = build_model(name, num_filters=16)
    path = str(tmp_path / "model.keras")
    model.save(path)
    restored = keras.models.load_model(path, compile=False)
    x = np.random.rand(1, 32, 32, 3).astype("float32")
    np.testing.assert_allclose(model(x), restored(x), atol=1e-6)


def test_tflite_conversion(tmp_path):
    litert = pytest.importorskip("ai_edge_litert.interpreter")
    from converter import convert

    model = build_model("SplitterNet", input_shape=(64, 64, 3), num_filters=16)
    path = str(tmp_path / "model.tflite")
    convert(model, path)

    interpreter = litert.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    x = np.random.rand(1, 64, 64, 3).astype("float32")
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    np.testing.assert_allclose(y, model(x), atol=1e-4)
