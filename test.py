import ModelSelector as ms
from os import path
import pickle

if __name__=='__main__':

    selector = ms.ModelSelector(path.join('.','data','const data test'), 
                                ms.base_models, ms.input_shape, 
                                save_model_path = path.join('.', 'const_models' , 'saved_models'))

    selector.load_models(load_from_local=False)

    selector.train_models()

    selector.test_models()

    with open('final_res', 'wb') as file:
        pickle.dump(selector.summary,file)

    with open('final_res','rb') as file:
        summary = pickle.load(file)

    print(summary)
    