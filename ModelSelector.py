from os import path
import tensorflow as tf
from CustomModel import CustomModel, BaseModel, predict_class
from CustomDirectoryIterator import CustomDirectoryIterator

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

class ModelSelector:
  def __init__(self, data_set_path, base_models, input_shape, epochs = 10) -> None:
    self.model_inp = list()
    self.models = dict()
    self.data_path = data_set_path
    self.acc = dict()
    for key in base_models.keys():
      model = dict()
      model["base_model_path"] = str(path.join("base_models", key))
      model["img_size"] = input_shape[key]
      model["epochs"] = epochs
      model["save_model_path"] = str(path.join("const models","saved_models", key))
      model["input_shape"] = input_shape[key]
      self.model_inp.append(model)

  def train_model(self, model: CustomModel, data_iterator: CustomDirectoryIterator, epoch, save_model_path):
    data, label = data_iterator.next()
    while data is not None:
      print(data_iterator.train_iterations)
      model.fit(data, label, epochs=epoch)
      data, label = data_iterator.next()
    model.save(save_model_path)
    self.models[save_model_path] = model
    print("completed training", save_model_path)

  def run_concurrently(self, load_from_local = True):
    self.iterators = dict()
    i = 0
    for temp in self.model_inp:
      self.iterators[temp["save_model_path"]] = CustomDirectoryIterator(self.data_path, (self.model_inp[i]["img_size"][0],self.model_inp[i]["img_size"][1]), 32, training_size=0.2)
      i+=1

    for i in range(len(self.model_inp)):
      if load_from_local == False:
        model = CustomModel(
          self.model_inp[i]["base_model_path"],
          self.model_inp[i]["save_model_path"],
          len(self.iterators[self.model_inp[i]["save_model_path"]].classes),
          self.model_inp[i]["input_shape"]
          )
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=tf.metrics.CategoricalAccuracy())
        self.train_model(model,self.iterators[self.model_inp[i]["save_model_path"]],self.model_inp[i]["epochs"], self.model_inp[i]["save_model_path"])
      else:
        self.models[self.model_inp[i]["save_model_path"]] = tf.keras.models.load_model(self.model_inp[i]["save_model_path"])
    print("completed running all the models")

  def accuracy(self, y, y_pred):
    c=0
    for i in range(len(y)):
      if y_pred[i]==y[i]:
        c+=1
    return c/len(y)
  
  def test_model(self, model: CustomModel, iterator: CustomDirectoryIterator):
    temp, count = 0, 0
    x,y = iterator.test_next()
    while x is not None:
      temp+=self.accuracy(y,predict_class(model,x))
      x,y = iterator.test_next()
      count+=1
    return temp/count

  def test(self):
    max_acc,out_key = 0,""
    for key in self.models:
      itr = self.iterators[key]
      model = self.models[key]
      val = self.test_model(model, itr)
      self.acc[key] = val
      if val>max_acc:
        max_acc=val
        out_key = key
    self.models[out_key].save("output_model/"+out_key)


  def download_basemodels(self):
    BaseModel(base_models["efficientnet_b7"], input_shape["efficientnet_b7"], "./base_models/efficientnet_b7")
    BaseModel(base_models["efficientnetv2_b3_21k_ft1k"], input_shape["efficientnetv2_b3_21k_ft1k"], "./base_models/efficientnetv2_b3_21k_ft1k")
    BaseModel(base_models["efficientnetv2_xl_21k_ft1k"], input_shape["efficientnetv2_xl_21k_ft1k"], "./base_models/efficientnetv2_xl_21k_ft1k")
    BaseModel(base_models["inception_resnet_v2"], input_shape["inception_resnet_v2"], "./base_models/inception_resnet_v2")
    BaseModel(base_models["mobilenet_v3_large_100_224"], input_shape["mobilenet_v3_large_100_224"], "./base_models/mobilenet_v3_large_100_224")