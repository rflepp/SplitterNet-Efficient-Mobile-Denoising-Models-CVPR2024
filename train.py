import os
import logging
import sys
import tensorflow as tf
import json
from dataloader import get_datasets
from models import Dynamic_PlainNet, Dynamic_UNet_simple, Megvii, MoDeNet, NOAHTCV, PlainNet, SplitterNet, ResNet_18
from utils import PSNRMetric, SSIMMetric, PSNRLoss, PrintLearningRate
from evaluate import evaluate_saved_model

def run(model_name, dir_path, epochs, batch_size, enc_blocks, dec_blocks, dataset_path, test_set_dir, filter_exp=5, saved_model_path=None):
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Starting.")
    
    logging.debug(f"Epochs: {epochs}, Batch Size: {batch_size}, Filters: {2**int(filter_exp)}, Saved model path: {saved_model_path}, "
                f"Dataset: {dataset_path}, Encoder Block Number: {enc_blocks}, Decoder Block Number: {dec_blocks}")

    ####################################################################################################################################
    # Data Loading
    ####################################################################################################################################

    TRAIN_ORIGINAL_PATCHES_PATH = dataset_path
    TRAIN_DENOISED_PATCHES_PATH = dataset_path

    CHECKPOINT_PATH = dir_path+"/checkpoints/"
    TRAINED_MODEL_PATH = dir_path+"/trained_model/"
    
    logging.debug("Loading Data")
    train_dataset, val_dataset = get_datasets(TRAIN_ORIGINAL_PATCHES_PATH, TRAIN_DENOISED_PATCHES_PATH, batch_size, val_split=0.1)

    ####################################################################################################################################
    # Model
    ####################################################################################################################################

    tf.keras.backend.clear_session()
    tf.random.set_seed(123)

    input_size = (None,None,3)
    model = None

    decay_steps = len(train_dataset)*int(epochs)
    initial_learning_rate = 4e-5
    final_learning_rate = 7e-06

    lr_schedule = tf.keras.experimental.CosineDecay(
        initial_learning_rate, decay_steps, alpha=final_learning_rate/initial_learning_rate)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999)

    if saved_model_path is not None and os.path.exists(saved_model_path):
        logging.debug("Loading Saved Model")
        custom_objects = {
            'PSNRMetric': lambda: PSNRMetric(max_val=1.0),
            'PSNRLoss': PSNRLoss(max_val=1.0),
        }

        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(saved_model_path, compile=False)
    else:
        logging.debug("Building Model")
        if model_name == "Dynamic_PlainNet":
            model = Dynamic_PlainNet.DYNUnet(input_size, enc_blocks, dec_blocks, 1, 2**int(filter_exp))
        elif model_name == "Dynamic_UNet_simple":
            model = Dynamic_UNet_simple.DYNUnet(input_size, enc_blocks, dec_blocks, 1, 2**int(filter_exp))
        elif model_name == "Megvii":
            model = Megvii.DYNUnet(input_size, enc_blocks, dec_blocks, 2**int(filter_exp))
        elif model_name == "MoDeNet":
            model = MoDeNet.DYNUnet(input_size, enc_blocks, dec_blocks, 1, 2**int(filter_exp))
        elif model_name == "NOAHTCV":
            model = NOAHTCV.Unet(input_size, 2**int(filter_exp))
        elif model_name == "PlainNet":
            model = PlainNet.DYNUnet(input_size, enc_blocks, dec_blocks, 1, 2**int(filter_exp))
        elif model_name == "SplitterNet":
            model = SplitterNet.DYNUnet(input_size, 2**int(filter_exp))
        elif model_name == "ResNet_18":
            model = ResNet_18.ResNet18_Denoiser(input_size)
    
    model.summary()

    loss = PSNRLoss()
    psnr_metric = PSNRMetric(max_val=1.0)
    ssim_metric = SSIMMetric(max_val=1.0)

    modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(CHECKPOINT_PATH, "model_{epoch:02d}.h5"),
                        monitor='val_ssim_metric',
                        save_best_only=False,
                        save_weights_only=False,
                        mode='max',
                        verbose=1)

    callbacks = [
        modelcheckpoint,
        PrintLearningRate(),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=[psnr_metric, ssim_metric, 'mean_squared_error'])


    train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    model.fit(train_dataset, epochs=int(epochs), callbacks=callbacks, validation_data=val_dataset)

    logging.debug('Saving Model.')
    model.save(TRAINED_MODEL_PATH, overwrite=True)

    ####################################################################################################################################
    # Evaluation
    ####################################################################################################################################
    
    logging.debug('Evaluating model')
    snr_values_denoised, ssim_values_denoised = evaluate_saved_model(model, test_set_dir)

    logging.debug("SSIM values of denoised images: %s",ssim_values_denoised)
    logging.debug("PSNR values of denoised images: %s",snr_values_denoised)

    logging.debug('Finishing.')

if __name__ == "__main__":
    model_name = sys.argv[1]
    epochs = sys.argv[2]
    batch_size = sys.argv[3]
    dir_path = sys.argv[4]
    enc_blocks = json.loads(sys.argv[5])
    dec_blocks = json.loads(sys.argv[6])
    dataset_path = sys.argv[7]
    test_set_dir = sys.argv[8]
    filter_exp = sys.argv[9]
    trained_path = sys.argv[10]
    run(model_name, dir_path, epochs, batch_size, enc_blocks, dec_blocks, dataset_path, test_set_dir, filter_exp, trained_path)
