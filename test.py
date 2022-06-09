from sklearn.metrics import accuracy_score
import ModelSelector as ms
from os import path

if __name__=='__main__':

    models = dict()
    
    key, val = ms.base_models.popitem()
    models[key] = val
    key, val = ms.base_models.popitem()
    models[key] = val
    selector = ms.ModelSelector('./data/const data test', models, ms.input_shape, save_model_path= './const_models/saved_models')

    selector.load_models()
    y_true, y_pred = selector.predict(path.join('.','data','const data test'))
    
    print(accuracy_score(y_true, y_pred))
    
    selector.test_models()

    print(selector.acc)