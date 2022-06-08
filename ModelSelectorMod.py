from genericpath import exists
from os import path
import tensorflow as tf
from CustomModel import CustomModel, BaseModel, predict_class
from CustomDirectoryIterator import CustomDirectoryIterator
from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, recall_score
import numpy as np

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
    self.acc = dict()             # Contains the accuracy of each model after slef.test() is called
    self.keys = list()            # Contains the model names

    if type == 'multiclass':
      self.activations = tf.nn.softmax
    elif type == 'multilabel':
      self.activations = tf.nn.sigmoid

    for key in base_models.keys():
      temp = dict()
      temp["base_model_path"] = str(path.join(base_model_path, "base_models", key))
      temp["img_size"] = input_shape[key]
      temp["save_model_path"] = str(path.join(save_model_path, key))
      temp["input_shape"] = input_shape[key]
      self.model_inp.append(temp)
      self.keys.append(key)
    
  def __init_iterators(self , batch_size, training_size):
    self.iterators = dict()
    for temp in self.model_inp:
      self.iterators[temp["save_model_path"]] = CustomDirectoryIterator(self.data_path, (temp["img_size"][0],temp["img_size"][1]), batch_size, training_size=training_size)
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
    while data is not None:
      print('Number of iterations remaining: ', data_iterator.train_iterations)
      model.fit(data, label, epochs=epoch)
      data, label = data_iterator.next()
      count_iter+=1
      if count_iter>check_point_iter:
        count_iter=0
        model.save(save_model_path)
    model.save(save_model_path)
    self.models[save_model_path] = model
    print("Completed training", save_model_path)

  def train_models(self, epochs = 10):
    for i in range(len(self.model_inp)):
      print("Trainig model ", i+1,'/',len(self.model_inp), end='\n')
      self.__train_model(self.models[self.model_inp[i]["save_model_path"]],
                      self.iterators[self.model_inp[i]["save_model_path"]],
                      epochs,
                      self.model_inp[i]["save_model_path"])
  
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
      self.acc[key] = val
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

  def predict(self, path):
    #create a custom iterator but resize according to size but for each image
    iterator = CustomDirectoryIterator(path, (300,300), training_size = 1.0, batch_size=32)
    input_shapes_predict = []
    i = 0
    model_size_dict = dict()
    for m in self.model_inp:
      model_size_dict[m["save_model_path"]] = i
      i += 1
      input_shapes_predict.append((m["input_shape"][0], m["input_shape"][1]))
    x = iterator.predict_next(input_shapes_predict)
    predicted = []
    while x is not None:
      preds = []
      for m in self.models:
        index = model_size_dict[m]
        preds.append(predict_class(self.models[m], np.array(x[index])))

      #voting
      for i in range(len(preds[0])):
        votes = dict()
        for j in range(len(preds)):
          if preds[j][i] in votes:
            votes[preds[j][i]] += 1
          else:
            votes[preds[j][i]] = 1
        max_votes = None
        vote_class = None
        for v in votes:
          if max_votes is None:
            max_votes = votes[v]
            vote_class = v
          else:
            if votes[v] > max_votes:
              max_votes = votes[v]
              vote_class = v
        predicted.append(vote_class)
      x = iterator.predict_next(input_shapes_predict)     
    print("completed")
    return predicted

  def download_basemodels(self):
    BaseModel(base_models["efficientnet_b7"], input_shape["efficientnet_b7"], "./base_models/efficientnet_b7")
    BaseModel(base_models["efficientnetv2_b3_21k_ft1k"], input_shape["efficientnetv2_b3_21k_ft1k"], "./base_models/efficientnetv2_b3_21k_ft1k")
    BaseModel(base_models["efficientnetv2_xl_21k_ft1k"], input_shape["efficientnetv2_xl_21k_ft1k"], "./base_models/efficientnetv2_xl_21k_ft1k")
    BaseModel(base_models["inception_resnet_v2"], input_shape["inception_resnet_v2"], "./base_models/inception_resnet_v2")
    BaseModel(base_models["mobilenet_v3_large_100_224"], input_shape["mobilenet_v3_large_100_224"], "./base_models/mobilenet_v3_large_100_224")
