import argparse
import os
import tensorflow as tf
from models.model import MSCC_end2end, MSCC_easy
from utils.dataset import loadCifarDataset
from utils.config import config

def main(args):
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load CIFAR-10 dataset
    train_ds, test_ds = loadCifarDataset(
        args.train_bs,
        args.test_bs,
        args.data_root,
        type='all',
        mini=args.mini
    )

    EXPERIMENT_NAME = args.experiment_name
    print(f'Running {EXPERIMENT_NAME}')

    model = MSCC_end2end(
        snr_db=         args.train_snrdB,
        num_tx=         args.num_tx,
        num_rx=         args.num_rx,
        channel_type=   args.channel_type,
        equalizer_type= args.equalizer_type,
        img_shape=      args.image_shape,
        channel_dim=    args.channel_dimension,
        ps=             args.patch_size,
        num_blocks=     args.num_blocks_in_layer,
        num_heads=      args.num_mhsa_heads,
        mlp_ratio=      args.mlp_ratio
    )

    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1)

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=args.learning_rate
        ),
        metrics=[
            psnr
        ]
    )

    model(tf.zeros([1, 32, 32, 3]))
    # model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    if args.ckpt is not None:
        model.load_weights(args.ckpt)

    save_ckpt = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"./logs/{EXPERIMENT_NAME}/epoch_" + "{epoch}",
            save_best_only=True,
            monitor="val_loss",
            save_weights_only=True,
            options=tf.train.CheckpointOptions(
                experimental_io_device=None, experimental_enable_async_checkpoint=True
            )
        )
    ]

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{EXPERIMENT_NAME}')
    history = model.fit(
        train_ds,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        callbacks=[tensorboard, save_ckpt],
        validation_data=test_ds,
    )

    model.save_weights(f"logs/{EXPERIMENT_NAME}/epoch_{args.epochs}")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help='experiment name (used for ckpt & logs)')
    parser.add_argument('--train_snrdB', type=int, default=config.TRAIN_SNR, help='train snr (in dB)')
    parser.add_argument('--learning_rate', type=int, default=config.LR, help='learning rate')
    
    parser.add_argument('--gpu', type=str, default=None, help='GPU index to use (e.g., "0" or "0,1")')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file path (optional)')
    parser.add_argument('--initial_epoch', type=int, default=0, help='initial epoch')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--mini', action='store_true', help='use mini dataset (with 100 images each)')
    
    parser.add_argument('--train_bs', type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument('--test_bs', type=int, default=config.TEST_BATCH_SIZE)
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT)
    
    parser.add_argument('--num_tx', type=int, default=1)
    parser.add_argument('--num_rx', type=int, default=1)
    parser.add_argument('--channel_type', type=str, default='flat')
    parser.add_argument('--equalizer_type', type=str, default='zf')
    
    parser.add_argument('--image_shape', type=tuple, default=config.IMAGE_SHAPE)
    parser.add_argument('--channel_dimension', type=int, default=config.CHANNEL_DIMENSION)
    parser.add_argument('--patch_size', type=int, default=config.PATCH_MERGE_SIZE)
    parser.add_argument('--num_blocks_in_layer', type=tuple, default=config.NUM_BLOCKS_IN_LAYER)
    parser.add_argument('--num_mhsa_heads', type=int, default=config.NUM_MHSA_HEADS)
    parser.add_argument('--mlp_ratio', type=int, default=config.MLP_RATIO)

    args = parser.parse_args()
    main(args)