import tensorflow as tf
import tensorflow_hub as hub
from keras import Model


"""
BaseModel
    Downloads base model(tf hub pretrained model) and save it in base_models directory at local machine
"""
class BaseModel(Model):
    def __init__(self, 
                handle, 
                input_shape, 
                save_model_path, 
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
                activation = tf.nn.softmax,
                optimizer = tf.keras.optimizers.Adam(),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics = tf.metrics.CategoricalAccuracy()
                ) -> None:
        super(CustomModel, self).__init__()
        self.base_layer = tf.keras.models.load(base_model_path)
        self.classifier = (tf.keras.layers.Dense(number_of_classes, activation = activation))
        self.build(self.base_layer.input_shape)
        self.compute_output_shape(input_shape=self.base_layer.input_shape)
        self.compile(optimizer=optimizer, loss = loss, metrics=metrics)
        self.save(save_model_path)

    def call(self, inputs, training=None, mask=None):
        return self.classifier(self.base_layer(inputs))