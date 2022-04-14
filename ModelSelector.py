from CustomModel import BaseModel, CustomModel
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


# TODO
class ModelSelector:
    def __init__(self, data_set_path) -> None:

        pass

if __name__=="__main__":
    BaseModel(base_models["efficientnet_b7"], input_shape["efficientnet_b7"], "./base_models/efficientnet_b7")
    BaseModel(base_models["efficientnetv2_b3_21k_ft1k"], input_shape["efficientnetv2_b3_21k_ft1k"], "./base_models/efficientnetv2_b3_21k_ft1k")
    BaseModel(base_models["efficientnetv2_xl_21k_ft1k"], input_shape["efficientnetv2_xl_21k_ft1k"], "./base_models/efficientnetv2_xl_21k_ft1k")
    BaseModel(base_models["inception_resnet_v2"], input_shape["inception_resnet_v2"], "./base_models/inception_resnet_v2")
    BaseModel(base_models["mobilenet_v3_large_100_224"], input_shape["mobilenet_v3_large_100_224"], "./base_models/mobilenet_v3_large_100_224")