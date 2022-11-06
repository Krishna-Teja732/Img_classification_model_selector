import ModelSelector as ms
import os
import pickle

if __name__=='__main__':

    mobi = ms.base_models.popitem()
    effixl = ms.base_models.popitem()

    models = dict()
    models[effixl[0]] = effixl[1]
    selector = ms.ModelSelector(os.path.join('.','data','Minet 5640 Images'), 
                                models, ms.input_shape, 
                                save_model_path = './minet_bal')

    selector.load_models(load_from_local=True, batch_size=512, training_size=1)

    selector.train_models()

    selector = ms.ModelSelector(os.path.join('.','data','Minet 5640 test'), 
                                models, ms.input_shape, 
                                save_model_path = './minet_bal')

    selector.load_models(load_from_local=True, training_size=0)

    selector.test_models()

    print(selector.summary)

    with open('./results/minet_bal_res/effixl.pkl', 'wb') as file:
        pickle.dump(selector.summary,file)