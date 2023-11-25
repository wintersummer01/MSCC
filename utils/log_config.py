import json

model_parameters = {
    'no_channel': {
        'train_class':'MSCC_easy',
        'train_params':{
            'img_shape'  : (32, 32, 3),
            'channel_dim': 256,
            'ps'         : 2,
            'num_blocks' : (2, 6),
            'num_heads'  : 8,
            'mlp_ratio'  : 4
        },
        'valid_params':{
            'snr_db'        : 0,
            'num_tx'        : 1,
            'num_rx'        : 1, 
            'channel_type'  : 'flat',
            'equalizer_type': 'zf',
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':185,
        'caption':'w/o channel (zf)'
    }, 
    
    'SISO_awgn_train10': {
        'train_class':'MSCC_end2end',
        'train_params':{
            'snr_db'        : 10,
            'num_tx'        : 1,
            'num_rx'        : 1, 
            'channel_type'  : 'awgn',
            'equalizer_type': None,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'valid_params':{
            'snr_db'        : 0,
            'num_tx'        : 1,
            'num_rx'        : 1, 
            'channel_type'  : 'flat',
            'equalizer_type': 'zf',
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':193,
        'caption':'awgn with SNR = 10dB (zf)'
    }, 
    
    'SISO_awgn_train20': {
        'train_class':'MSCC_end2end',
        'train_params':{
            'snr_db'        : 20,
            'num_tx'        : 1,
            'num_rx'        : 1, 
            'channel_type'  : 'awgn',
            'equalizer_type': None,
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'valid_params':{
            'snr_db'        : 0,
            'num_tx'        : 1,
            'num_rx'        : 1, 
            'channel_type'  : 'flat',
            'equalizer_type': 'zf',
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':194,
        'caption':'awgn with SNR = 20dB (zf)'
    },
    
    'SISO_zf_train10': {
        'train_class':'MSCC_end2end',
        'train_params':{
            'snr_db'        : 10,
            'num_tx'        : 1,
            'num_rx'        : 1, 
            'channel_type'  : 'flat',
            'equalizer_type': 'zf',
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'valid_params':{
            'snr_db'        : 0,
            'num_tx'        : 1,
            'num_rx'        : 1, 
            'channel_type'  : 'flat',
            'equalizer_type': 'zf',
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':173,
        'caption':'Rayleigh with SNR = 10dB (zero forcing)'
    }, 
    
    'SISO_dnn_train10': {
        'train_class':'MSCC_end2end',
        'train_params':{
            'snr_db'        : 10,
            'num_tx'        : 1,
            'num_rx'        : 1, 
            'channel_type'  : 'flat',
            'equalizer_type': 'dnn',
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'valid_params':{
            'snr_db'        : 0,
            'num_tx'        : 1,
            'num_rx'        : 1, 
            'channel_type'  : 'flat',
            'equalizer_type': 'dnn',
            'img_shape'     : (32, 32, 3),
            'channel_dim'   : 256,
            'ps'            : 2,
            'num_blocks'    : (2, 6),
            'num_heads'     : 8,
            'mlp_ratio'     : 4
        },
        'epoch':497,
        'caption':'Rayleigh with SNR = 10dB (transformer)'
    }
}

with open('logs/model_configs.json', 'w') as f:
    json.dump(model_parameters, f)
    