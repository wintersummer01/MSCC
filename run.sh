CUDA_VISIBLE_DEVICES=0 nohup python train.py no_channel --epochs 50 &> logs/nohup/no_channel.out &

CUDA_VISIBLE_DEVICES=0 nohup python train.py SISO_awgn_train10 --train_snrdB 10 --epochs 200 --channel_type awgn &> logs/SISO_awgn_train10/nohup.out &
CUDA_VISIBLE_DEVICES=1 nohup python train.py SISO_awgn_train20 --train_snrdB 20 --epochs 200 --channel_type awgn &> logs/SISO_awgn_train20/nohup.out &

CUDA_VISIBLE_DEVICES=0 nohup python train.py SISO_zf_train10 --train_snrdB 10 --epochs 200 --equalizer_type zf &> logs/SISO_zf_train10/nohup.out &
CUDA_VISIBLE_DEVICES=1 nohup python train.py SISO_dnn_train10 --train_snrdB 10 --epochs 200 --equalizer_type dnn &> logs/SISO_dnn_train10/nohup.out &

python valid.py --valid_type sample_img
python valid.py --valid_type snr_graph --mini --test_bs 256

CUDA_VISIBLE_DEVICES=1 nohup python train.py SISO_dnn_train10 --train_snrdB 10 --epochs 500 --equalizer_type dnn --initial_epoch 200 --ckpt logs/SISO_dnn_train10/epoch_199 &> logs/SISO_dnn_train10/nohup.out &