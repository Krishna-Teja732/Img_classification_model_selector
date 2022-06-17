from sklearn.metrics import accuracy_score
from pprint import pprint
import ModelSelector as ms
from os import path

if __name__=='__main__':

    models = dict()
    
    key, val = ms.base_models.popitem()
    models[key] = val
    selector = ms.ModelSelector(path.join('.','data','const data test'), 
                                models, ms.input_shape, 
                                save_model_path= path.join('.', 'const_models' , 'saved_models'))

    selector.load_models(load_from_local=False)

    selector.train_models(tune_hyperparameters=True, max_trials= 2)

    selector.test_models()

    selector.predict(path.join('.','data','const data test'), path.join('.','data','saved_images'))
    

    pprint(selector.summary)

    