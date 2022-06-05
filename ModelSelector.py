from os import path
import tensorflow as tf
from CustomModel import CustomModel, BaseModel, predict_class
from CustomDirectoryIterator import CustomDirectoryIterator
from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, recall_score
import numpy as np


# base_models = {
#     "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
#     "efficientnetv2_b3_21k_ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
#     "efficientnetv2_xl_21k_ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
#     "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5",
#     "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
# }

# input_shape = {
#     "efficientnet_b7": (600,600,3),
#     "efficientnetv2_b3_21k_ft1k": (300,300,3),
#     "efficientnetv2_xl_21k_ft1k": (512,512,3),
#     "inception_resnet_v2": (299,299,3),
#     "mobilenet_v3_large_100_224": (224,224,3)
# }

base_models = {
    "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1"
}

input_shape = {
    "efficientnet_b7": (600,600,3)
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
              ) -> None:
    self.model_inp = list()       # Contains inputs that are required to load the model
    self.models = dict()          # Contains tf.keras.models as values
    self.data_path = dataset_path # ~~
    self.acc = dict()             # Contains the accuracy of each model after slef.test() is called
    self.keys = list()            # Contains the model names

    for key in base_models.keys():
      temp = dict()
      temp["base_model_path"] = str(path.join("base_models", key))
      temp["img_size"] = input_shape[key]
      temp["save_model_path"] = str(path.join(save_model_path, key))
      temp["input_shape"] = input_shape[key]
      self.model_inp.append(temp)
      self.keys.append(key)
    
  # function used to train a single model
  def train_model(self, model: CustomModel, data_iterator: CustomDirectoryIterator, epoch, save_model_path):
    data, label = data_iterator.next()
    while data is not None:
      print('number of iterations remaining: ', data_iterator.train_iterations)
      model.fit(data, label, epochs=epoch)
      data, label = data_iterator.next()
    model.save(save_model_path)
    self.models[save_model_path] = model
    print("completed training", save_model_path)
  
  def load_models(self, load_from_local = True):
    self.iterators = dict()
    i = 0
    for temp in self.model_inp:
      self.iterators[temp["save_model_path"]] = CustomDirectoryIterator(self.data_path, (self.model_inp[i]["img_size"][0],self.model_inp[i]["img_size"][1]), 32, training_size=0.2)
      i+=1

    for i in range(len(self.model_inp)):
      print("\rLoading model ", i+1,'/',len(self.model_inp), end='')
      if load_from_local == False:
        model = CustomModel(
          self.model_inp[i]["base_model_path"],
          self.model_inp[i]["save_model_path"],
          len(self.iterators[self.model_inp[i]["save_model_path"]].classes),
          self.model_inp[i]["input_shape"]
          )
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=tf.metrics.CategoricalAccuracy())
      else:
        model = tf.keras.models.load_model(self.model_inp[i]["save_model_path"])
      self.models[self.model_inp[i]["save_model_path"]] = model

  def train_models(self):
    for i in range(len(self.model_inp)):
      print("\rTrainig model ", i+1,'/',len(self.model_inp), end='')
      self.train_model(self.models[self.model_inp[i]["save_model_path"]],
                      self.iterators[self.model_inp[i]["save_model_path"]],
                      10,
                      self.model_inp[i]["save_model_path"])
  
  def accuracy(self, y, y_pred):
    c=0
    for i in range(len(y)):
      if y_pred[i]==y[i]:
        c+=1
    return c/len(y)

  def test(self):
    max_acc,out_key = 0,""
    for key in self.models:
      itr = self.iterators[key]
      model = self.models[key]
      val = self.test_model_mod(model, itr)
      self.acc[key] = val
      val = val["accuracy"]
      if val>max_acc:
        max_acc=val
        out_key = key
    self.models[out_key].save("output_model/"+out_key)

  def test_model(self, model: CustomModel, iterator: CustomDirectoryIterator):
    temp, count = 0, 0
    x,y = iterator.test_next()
    while x is not None:
      temp+=self.accuracy(y,predict_class(model,x))
      x,y = iterator.test_next()
      count+=1
    return temp/count

  def test_model_mod(self, model: CustomModel, iterator: CustomDirectoryIterator):
    y_pred, y_test = list(), list()
    x,y = iterator.test_next()
    while x is not None:
      y_pred.extend(predict_class(model,x))
      y_test.extend(y)
      x,y = iterator.test_next()
    # RocCurveDisplay.from_predictions(y_test, y_pred, y_type = 'multiclass')
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
