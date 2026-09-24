"""Train a denoising model.

Example:
    python train.py --model SplitterNet --epochs 20 --batch-size 16 --output-dir runs/splitternet \\
        --dataset path/to/train/patches --test-dir path/to/test_set
"""
import argparse
import logging
import os

# TensorFlow's oneDNN CPU kernels segfault when back-propagating through the 1x1, stride-2
# Conv2DTranspose layers of MoDeNet/NOAHTCV (TF 2.21). This only affects CPU training.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras  # noqa: E402

from dataloader import get_datasets  # noqa: E402
from evaluate import evaluate_model, load_model  # noqa: E402
from models import MODEL_NAMES, build_model  # noqa: E402
from utils import PrintLearningRate, PSNRLoss, PSNRMetric, SSIMMetric  # noqa: E402

logger = logging.getLogger(__name__)

INITIAL_LEARNING_RATE = 4e-5
FINAL_LEARNING_RATE = 7e-6


def parse_blocks(value):
    """Parse a block configuration such as ``1,1,1,1`` or ``[2,2,4,8]``."""
    return [int(v) for v in value.strip("[]").split(",")]


def train(model_name, output_dir, epochs, batch_size, dataset, test_dir=None, filter_exp=5,
          enc_blocks=(1, 1, 1, 1), dec_blocks=(1, 1, 1, 1), checkpoint=None):
    num_filters = 2 ** filter_exp
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset, val_dataset = get_datasets(dataset, batch_size, val_split=0.1)

    keras.utils.set_random_seed(123)
    if checkpoint:
        logger.info("Loading %s", checkpoint)
        model = load_model(checkpoint, model_name, num_filters)
    else:
        model = build_model(model_name, num_filters=num_filters, enc_blocks=enc_blocks, dec_blocks=dec_blocks)
    model.summary()

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        INITIAL_LEARNING_RATE,
        decay_steps=int(train_dataset.cardinality()) * epochs,
        alpha=FINAL_LEARNING_RATE / INITIAL_LEARNING_RATE,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=PSNRLoss(max_val=1.0),
        metrics=[PSNRMetric(max_val=1.0), SSIMMetric(max_val=1.0), "mean_squared_error"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, "model_{epoch:02d}.keras"), verbose=1),
        PrintLearningRate(),
    ]
    model.fit(train_dataset, epochs=epochs, callbacks=callbacks, validation_data=val_dataset)

    model_path = os.path.join(output_dir, "trained_model.keras")
    logger.info("Saving model to %s", model_path)
    model.save(model_path)

    if test_dir:
        evaluate_model(model, test_dir)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--dataset", required=True, help="Training data directory (see dataloader.py for supported layouts)")
    parser.add_argument("--output-dir", required=True, help="Where checkpoints and the trained model are written")
    parser.add_argument("--test-dir", help="Optional test set evaluated after training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--filter-exp", type=int, default=5, help="Number of filters = 2**filter_exp (default: 5)")
    parser.add_argument("--enc-blocks", type=parse_blocks, default=[1, 1, 1, 1], help="Blocks per encoder stage, e.g. 1,1,1,1")
    parser.add_argument("--dec-blocks", type=parse_blocks, default=[1, 1, 1, 1], help="Blocks per decoder stage, e.g. 1,1,1,1")
    parser.add_argument("--checkpoint", help="Resume from a .keras model or initialise from a .h5 weights file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(args.model, args.output_dir, args.epochs, args.batch_size, args.dataset, args.test_dir,
          args.filter_exp, args.enc_blocks, args.dec_blocks, args.checkpoint)


if __name__ == "__main__":
    main()
