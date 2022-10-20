import json
import os
import shutil
import numpy as np
from os import path
import tensorflow as tf
from CustomTuner import CustomTuner
from CustomDirectoryIterator import CustomDirectoryIterator
from CustomModel import CustomModel, BaseModel, max_ind, predict_class
from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, recall_score

base_models = {
    "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
    "efficientnetv2_b3_21k_ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
    "efficientnetv2_xl_21k_ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
    "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5",
    "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
}

input_shape = {
    "efficientnet_b7": (600,600,3),
    "efficientnetv2_b3_21k_ft1k": (300,300,3),
    "efficientnetv2_xl_21k_ft1k": (512,512,3),
    "inception_resnet_v2": (299,299,3),
    "mobilenet_v3_large_100_224": (224,224,3)
}

"""
ModelSelector
----------------------------
This class gets an image dataset path and multiple Tensorflowhub 
pretrained models and trains those models on the given dataset.
The retrained models can be tested for classification accuracy
and the model with most accuracy is stored in output directory.
"""
class ModelSelector:
  def __init__(self,
              dataset_path: str,
              base_models: dict,
              input_shape: dict,
              save_model_path='.',
              base_model_path='.',
              type = 'multiclass'
              ) -> None:
    self.model_inp = list()       # Contains inputs that are required to load the model
    self.models = dict()          # Contains tf.keras.models as values
    self.data_path = dataset_path 
    self.summary = dict()             # Contains the accuracy of each model after slef.test() is called
    self.keys = list()            # Contains the model names

    if type == 'multiclass':
      self.activations = tf.nn.softmax
    elif type == 'multilabel':
      self.activations = tf.nn.sigmoid

    for key in base_models.keys():
      temp = dict()
      temp["base_model_path"] = str(path.join(base_model_path, "base_models", key))
      temp["save_model_path"] = str(path.join(save_model_path, key))
      temp["input_shape"] = input_shape[key]
      self.model_inp.append(temp)
      self.keys.append(key)
    
  def __init_iterators(self , batch_size, training_size):
    self.iterators = dict()
    for temp in self.model_inp:
      self.iterators[temp["save_model_path"]] = CustomDirectoryIterator(self.data_path, (temp["input_shape"][0],temp["input_shape"][1]), batch_size, training_size=training_size)
    self.out_dim = len(self.iterators[self.model_inp[0]["save_model_path"]].classes)

  def __init_custom_model(self, base_model_path, save_model_path, output_layer_len, input_shape, activation, optimizer)-> CustomModel:
    model = CustomModel(
          base_model_path,
          save_model_path,
          output_layer_len,
          input_shape,
          activation
          )
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=tf.metrics.CategoricalAccuracy())
    return model

  def load_models(self, load_from_local = True, batch_size=32, training_size=0.8, optimizer = 'adam'):
    self.__init_iterators(batch_size, training_size)
    count = 1
    for model_inp in self.model_inp:
      print("\rLoading model ", count,'/',len(self.model_inp), end='')
      if load_from_local == False:
        model = self.__init_custom_model(
          model_inp["base_model_path"],
          model_inp["save_model_path"],
          self.out_dim,
          model_inp["input_shape"],
          self.activations,
          optimizer
          )
      else:
        model = tf.keras.models.load_model(model_inp["save_model_path"])
      self.models[model_inp["save_model_path"]] = model
      count+=1

  # function used to train a single model
  def __train_model(self, model: CustomModel, data_iterator: CustomDirectoryIterator, epoch, save_model_path, check_point_iter = 10):
    data, label = data_iterator.next()
    count_iter = 1
    file = open(os.path.join(save_model_path, 'history.txt'), 'w')
    while data is not None:
      print('Number of iterations remaining: ', data_iterator.train_iterations)
      history = model.fit(data, label, epochs=epoch)
      file.write(json.dumps(history.history)+'\n')
      data, label = data_iterator.next()
      count_iter+=1
      if count_iter>check_point_iter:
        count_iter=0
        model.save(save_model_path)
    file.flush()
    file.close()
    model.save(save_model_path)
    self.models[save_model_path] = model
    print("Completed training ", save_model_path)

  def train_models(self, epochs = 10, tune_hyperparameters = False, max_trials = 10):
    for i in range(len(self.model_inp)):
      print("Trainig model ", i+1,'/',len(self.model_inp), end='\n')
      model = self.models[self.model_inp[i]["save_model_path"]]
      itr = self.iterators[self.model_inp[i]["save_model_path"]]
      save_path = self.model_inp[i]["save_model_path"]
      if not tune_hyperparameters:
        self.__train_model(model,itr,epochs,save_path)
      else:
        tuner = CustomTuner(save_path, itr, max_trials=max_trials)
        tuner.search()
        model = tuner.get_best_models(1)[0]
        model.save(save_path)
        self.models[save_path] = model
        shutil.rmtree(os.path.join('.', 'test_model_params'))

  def accuracy(self, y, y_pred):
    c=0
    for i in range(len(y)):
      if y_pred[i]==y[i]:
        c+=1
    return c/len(y)

  def test_models(self):
    max_acc,out_key = 0,""
    for key in self.models:
      itr = self.iterators[key]
      model = self.models[key]
      val = self.__test_model(model, itr)
      self.summary[key] = val
      val = val["accuracy"]
      if val>=max_acc:
        max_acc=val
        out_key = key
    assert out_key!="", 'Error in testing models, Check if Number of test iterations in Custom Iterator is nonzero'
    self.models[out_key].save("output_model/"+out_key)

  def __test_model(self, model: CustomModel, iterator: CustomDirectoryIterator):
    y_pred, y_test = list(), list()
    x,y = iterator.test_next()
    while x is not None:
      y_pred.extend(predict_class(model,x))
      y_test.extend(y)
      x,y = iterator.test_next()
    return {'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred, average = 'macro'), 
            'precesion': precision_score(y_test, y_pred, average = 'macro'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

  def predict(self, path, save_path, string_labels = True, overwrite = True):
    iterator = CustomDirectoryIterator(path, (300,300), training_size = 1.0, batch_size=32)
    input_sizes = [inp["input_shape"][:-1] for inp in self.model_inp]
    models = [self.models[inp["save_model_path"]] for inp in self.model_inp]
    x,y = iterator.predict_next(input_sizes)
    y_final = list()
    y_true = list()
    if os.path.exists(save_path) and overwrite:
      shutil.rmtree(save_path)
    os.mkdir(save_path)

    count = 1
    while x is not None:
      for image in x[0]:
        tf.keras.utils.save_img(os.path.join(save_path, str(count) + '.jpeg'), image.numpy())
        count+=1
      y_res = np.zeros((iterator.BATCH_SIZE, self.out_dim))
      for index in range(len(input_sizes)):
        y_pred = predict_class(models[index], np.array(x[index]))
        for img_index in range(len(y_pred)):
          y_res[img_index][y_pred[img_index]]+=1
      y_classes_batch = list(map(max_ind, y_res))
      y_final.extend(y_classes_batch)
      y_true.extend(y)
      x,y = iterator.predict_next(input_sizes)
    
    if string_labels:
      classes = self.iterators[list(self.iterators.keys())[0]].classes
      return list(map(lambda index: classes[index],y_true)), list(map(lambda index: classes[index],y_final))

    return y_true, y_final

  def download_basemodels(self):
    BaseModel(base_models["efficientnet_b7"], input_shape["efficientnet_b7"], "./base_models/efficientnet_b7")
    BaseModel(base_models["efficientnetv2_b3_21k_ft1k"], input_shape["efficientnetv2_b3_21k_ft1k"], "./base_models/efficientnetv2_b3_21k_ft1k")
    BaseModel(base_models["efficientnetv2_xl_21k_ft1k"], input_shape["efficientnetv2_xl_21k_ft1k"], "./base_models/efficientnetv2_xl_21k_ft1k")
    BaseModel(base_models["inception_resnet_v2"], input_shape["inception_resnet_v2"], "./base_models/inception_resnet_v2")
    BaseModel(base_models["mobilenet_v3_large_100_224"], input_shape["mobilenet_v3_large_100_224"], "./base_models/mobilenet_v3_large_100_224")
