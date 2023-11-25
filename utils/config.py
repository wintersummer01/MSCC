class config:
    # Useful roots/dirs
    DATA_ROOT = '/workspace/dataset/cifar-10-batches-py/'

    # Data Spec
    IMAGE_SHAPE = (32, 32, 3)
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 256

    # Training Spec
    LR = 1e-4
    TRAIN_SNR = 10

    # transformer Spec
    PATCH_MERGE_SIZE = 2
    MLP_RATIO = 4
    CHANNEL_DIMENSION = 256
    NUM_BLOCKS_IN_LAYER = (2, 6)
    NUM_MHSA_HEADS = 8