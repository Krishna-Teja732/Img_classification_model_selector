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

    selector.load_models()

    # selector.train_models(tune_hyperparameters=True, max_trials= 1)

    selector.test_models()

    # y_true, y_pred = selector.predict(path.join('.','data','const data test'), path.join('.','data','saved_images'))

    # for val,pred in zip(y_true, y_pred):
    #     print(val, pred)

    for key in selector.summary:
        print("Model: ", key)
        pprint(selector.summary[key])

    