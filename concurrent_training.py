from multiprocessing import Process
from image_classification.Img_classification_model_selector.CustomModel import CustomModel
from CustomDirectoryIterator import CustomDirectoryIterator

def train_model(model, data_iterator):
    print(model)

def run_concurrently(base_models, data_path):
    processes = []
    iterators = [CustomDirectoryIterator(data_path, base_models[i]["img_size"]) for i in range(len(base_models))]

    for i in range(base_models):
        model = CustomModel(
            base_models[i]["base_model_path"],
            base_models[i]["save_model_path"],
            base_models[i]["number_of_classes"],
            base_models[i]["activation"])
        process = Process(target=train_model, args=(model,iterators[i]))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()

    print("complete running all the models")
