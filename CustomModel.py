import tensorflow as tf
import tensorflow_hub as hub
from keras import Model
from multiprocessing import Pool

"""
BaseModel
    Downloads base model(tf hub pretrained model) and save it in base_models directory at local machine
"""
class BaseModel(Model):
    def __init__(self, 
                handle, 
                input_shape, 
                save_model_path
                ) -> None:
        super(BaseModel, self).__init__()
        self.base_layer = (hub.KerasLayer(handle,input_shape = input_shape))
        self.build((None,)+input_shape)
        self.compute_output_shape(input_shape=(None,)+input_shape)
        self.save(save_model_path)

    def call(self, inputs, training=None, mask=None):
        return self.base_layer(inputs)


"""
    CustomModel: adds classifier to base model, compiles it and saved it at the specified path(save_model_path)
    custom model can be loaded from the memory using keras.models.load(PATH) function
"""
class CustomModel(Model):
    def __init__(self,
                base_model_path, 
                save_model_path, 
                number_of_classes,
                input_shape,
                activation = tf.nn.softmax
                ) -> None:
        super(CustomModel, self).__init__()
        self.base_layer = tf.keras.models.load_model(base_model_path)
        self.classifier = (tf.keras.layers.Dense(number_of_classes, activation = activation))
        self.compute_output_shape(input_shape=((None,)+input_shape))
        self.save(save_model_path)

    def call(self, inputs, training=None, mask=None):
        return self.classifier(self.base_layer(inputs))

def max_ind(prob):
  max, val = 0, 0
  for i in range(len(prob)):
    if prob[i]>val:
      val = prob[i]
      max = i
  return max

def predict_class(model, x):
    pool = Pool()
    async_res = pool.map_async(max_ind, model.predict(x))
    return async_res.get()