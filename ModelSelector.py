from os import path
import tensorflow as tf
from multiprocessing import Process
from CustomModel import CustomModel, BaseModel
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

def train_model(model: CustomModel, data_iterator: CustomDirectoryIterator, epoch, save_model_path):
  data, label = data_iterator.next()
  count=0
  while data is not None:
    print(data_iterator.train_iterations)
    model.fit(data, label, epochs=epoch)
    data, label = data_iterator.next()
    count = count+1
  model.save(save_model_path)
  print("completed training", save_model_path)


def run_concurrently(base_models, data_path):
  processes = []
  iterators = [CustomDirectoryIterator(data_path, (base_models[i]["img_size"][0],base_models[i]["img_size"][1])) for i in range(len(base_models))]
  

  for i in range(len(base_models)):
      model = CustomModel(
          base_models[i]["base_model_path"],
          base_models[i]["save_model_path"],
          len(iterators[0].classes),
          base_models[i]["input_shape"]
          )
      model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=tf.metrics.CategoricalAccuracy())
      process = Process(target=train_model, args=(model,iterators[i],base_models[i]["epochs"], base_models[i]["save_model_path"]))
      processes.append(process)
      process.start()

  for p in processes:
      p.join()

  print("completed running all the models")


class ModelSelector:
  def __init__(self, data_set_path) -> None:
    models = list()
    global base_models
    for key in base_models.keys():
      model = dict()
      model["base_model_path"] = str(path.join("base_models", key))
      model["img_size"] = input_shape[key]
      model["epochs"] = 1
      model["save_model_path"] = str(path.join("saved_models", key))
      model["input_shape"] = input_shape[key]
      models.append(model)
    run_concurrently(models,data_set_path)

  def download_basemodels(self):
    BaseModel(base_models["efficientnet_b7"], input_shape["efficientnet_b7"], "./base_models/efficientnet_b7")
    BaseModel(base_models["efficientnetv2_b3_21k_ft1k"], input_shape["efficientnetv2_b3_21k_ft1k"], "./base_models/efficientnetv2_b3_21k_ft1k")
    BaseModel(base_models["efficientnetv2_xl_21k_ft1k"], input_shape["efficientnetv2_xl_21k_ft1k"], "./base_models/efficientnetv2_xl_21k_ft1k")
    BaseModel(base_models["inception_resnet_v2"], input_shape["inception_resnet_v2"], "./base_models/inception_resnet_v2")
    BaseModel(base_models["mobilenet_v3_large_100_224"], input_shape["mobilenet_v3_large_100_224"], "./base_models/mobilenet_v3_large_100_224")