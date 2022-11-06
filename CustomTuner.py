import keras_tuner
import tensorflow as tf
from sklearn.metrics import accuracy_score
from CustomModel import CustomModel, predict_class
from CustomDirectoryIterator import CustomDirectoryIterator

class CustomTuner(keras_tuner.RandomSearch):
    def __init__(self, model_path, iterator: CustomDirectoryIterator, max_trials = 10, seed = None) -> None:
        self.model_path = model_path
        self.iterator = iterator
        self.trial_models = dict()
        super(CustomTuner, self).__init__(max_trials= max_trials, seed= seed, project_name = 'test_model_params')
    
    def __train_model(self, model: CustomModel, data_iterator: CustomDirectoryIterator, epoch):
        data, label = data_iterator.next()
        while data is not None:
            print('Number of iterations remaining: ', data_iterator.train_iterations)
            model.fit(data, label, epochs=epoch)
            data, label = data_iterator.next()

    def __test_model(self, model: CustomModel, iterator: CustomDirectoryIterator):
        y_pred, y_test = list(), list()
        x,y = iterator.test_next()
        while x is not None:
            y_pred.extend(predict_class(model,x))
            y_test.extend(y)
            x,y = iterator.test_next()
        return accuracy_score(y_test, y_pred)

    def __get_optimizer(self, name, learning_rate):
        if name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate)
        elif name == 'adadelta':
            return tf.keras.optimizers.Adadelta(learning_rate)
        return tf.keras.optimizers.Nadam(learning_rate)

    def run_trial(self, trial):
        hp = trial.hyperparameters
        learning_rate = hp.Float("learning_rate", min_value = 1e-4, max_value = 1e-2)
        optimizer = self.__get_optimizer(hp.Choice("optimizer",["adam", "adadelta", "nadam"]), learning_rate)
        epoch = hp.Int("epochs", min_value = 10, max_value = 15)
        model = tf.keras.models.load_model(self.model_path)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=tf.metrics.CategoricalAccuracy())
        

        self.__train_model(model, self.iterator, epoch)

        self.trial_models[trial.trial_id] = model

        return -self.__test_model(model, self.iterator)

    def get_best_models(self, num_models=1):
        trials = self.oracle.get_best_trials(num_models)
        return [self.trial_models[trial.trial_id] for trial in trials]