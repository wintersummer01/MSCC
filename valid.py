import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import sionna as sn
from utils.load import loadModels
from utils.dataset import loadCifarDataset
from utils.config import config

def snr_graph(models, experiments, captions, test_ds):
    plot_marker = ['-o', '-v', '-s', '-d']
    
    SNRs = range(1, 26, 3)
    for i, model in enumerate(models):
        print(f"validating {experiments[i]}")
        meanPSNR = []
        for SNR in tqdm(SNRs):
            no = sn.utils.ebnodb2no(SNR, num_bits_per_symbol=1, coderate=1)
            PSNRs = []
            for img, _ in tqdm(test_ds, leave=False):
                output = model(img, no)
                PSNRs.append(tf.image.psnr(img, output, max_val=1))
            meanPSNR.append(np.mean(tf.concat(PSNRs, 0)))
        plt.plot(SNRs, meanPSNR, plot_marker[i], label=captions[i])
        plt.title(f'Various models in rayleigh channel')
        plt.legend()
        plt.xlabel('SNR_test (dB)')
        plt.ylabel('PSNR (dB)')
        plt.savefig('images/rayleigh.png')



def sample_img(models, experiments, test_ds, **kwargs):    
    img = tf.expand_dims(iter(test_ds).get_next()[0][11,:], axis=0)
    no = sn.utils.ebnodb2no(10, num_bits_per_symbol=1, coderate=1)
    _, axes = plt.subplots(2, 2, constrained_layout = True)
    axes = axes.ravel()
    
    for i, model in enumerate(models):
        output = model(img, no)
        psnr = tf.image.psnr(img, output, max_val=1)
        axes[i].imshow(tf.squeeze(output))
        axes[i].set_title(f'{experiments[i]}, (psnr = {psnr[0]:.2f})')
    axes[0].imshow(tf.squeeze(img))
    axes[0].set_title('original image')
    plt.savefig('images/sample_images.png')
        
        

def img_hist(models, **kwargs):
    img = mpimg.imread('images/hjyoo.png')[:,:,0:3]
    _, axes = plt.subplots(2, 2, constrained_layout = True)
    model = models[1]
    latent = tf.reshape(tf.squeeze(model.encoder(img)), [8, 8, 256])
    axes[0][0].imshow(tf.squeeze(img))
    axes[0][0].set_title('original image')
    axes[0][1].hist(tf.reshape(latent, [-1]), bins=40)
    axes[0][1].set_title('overall distribution')
    axes[1][0].imshow(tf.math.reduce_mean(latent, axis=2), cmap='gray')
    axes[1][0].set_title('mean of each vector')
    axes[1][1].imshow(tf.math.reduce_std(latent, axis=2), cmap='gray')
    axes[1][1].set_title('std dev of each vector')
    plt.savefig('images/image_hist.png')
    plt.clf()
    _, axes = plt.subplots(8, 8, constrained_layout = True)
    for i in range(8):
        for j in range(8):
            axes[i][j].hist(latent[i,j], bins=30)
            axes[i][j].xaxis.set_visible(False)
            axes[i][j].yaxis.set_visible(False)
    plt.savefig('images/latent_hist.png')
    


# Equalizer을 랜덤비트에 이퀄라이저만 넣어보기

# Std dev를 베이스로하는 Power allocation method
        
        

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('valid_type', type=str)
    parser.add_argument('--experiments', type=str, 
                        default=['no_channel', 'SISO_awgn_train10', 'SISO_zf_train10', 'SISO_dnn_train10'],
                        help='experiment names')
    parser.add_argument('--mini', action='store_true', help='use mini dataset (with 100 images each)')
    parser.add_argument('--test_bs', type=int, default=config.TEST_BATCH_SIZE)
    parser.add_argument('--data_root', type=str, default=config.DATA_ROOT)
    args = parser.parse_args()
    
    test_ds = loadCifarDataset(
        None,
        args.test_bs,
        args.data_root,
        type='test',
        mini=args.mini
    )
    models, captions = loadModels(args.experiments)
    valid_function = globals()[args.valid_type]
    
    valid_function(
        models     = models, 
        experiments= args.experiments, 
        captions   = captions, 
        test_ds    = test_ds[0]
    )