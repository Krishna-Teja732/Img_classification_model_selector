import os
import io
import PIL
from pathlib import Path
from PIL import Image as pil_image

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS

class ImageDataSetFilter:
    def __init__(self, data_set_path) -> None:
        self.data_set_path = data_set_path
        self.img_paths = list()
        self.img_classes_path = list(map(lambda class_name: os.path.join(data_set_path, class_name),os.listdir(data_set_path)))
        for class_path in self.img_classes_path:
            for record in os.walk(class_path):
                self.img_paths.extend(list(map(lambda x: os.path.join(class_path,x),record[2])))

    def check_dataset(self):
        for img in self.img_paths:
            try:
                with open(img , 'rb') as f:
                    pil_image.open(io.BytesIO(f.read()))    
                if pil_image.open(img).format not in ['JPEG', 'PNG']:
                    os.remove(Path(img))
            except PIL.UnidentifiedImageError:
                os.remove(Path(img))