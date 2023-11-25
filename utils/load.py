import json
import tensorflow as tf
from models.model import MSCC_end2end

def loadModel(model_name):
    with open('logs/model_configs.json', 'r') as f:
        model_configs = json.load(f)
    model_config = model_configs[model_name]
    model = MSCC_end2end(**model_config['valid_params'])
    model(tf.zeros([1, 32, 32, 3]))
    model.load_weights(f'./logs/{model_name}/epoch_{model_config["epoch"]}')
    return model, model_config["caption"]
    
def loadModels(model_names):
    models = []
    captions = []
    for model_name in model_names:
        model, caption = loadModel(model_name)
        models.append(model)
        captions.append(caption)
    return models, captions
    
    