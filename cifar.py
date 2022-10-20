import os
from uuid import uuid4
from PIL import Image
from keras.datasets import cifar100

def save_pngs(imgs, labs, base_path):
    tot,ind = len(labs), 0
    for img, lab in zip(imgs, labs):
        print("\rProgress ", (ind/tot)*100, end='')
        ind = ind+1
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img.save(os.path.join(base_path, str(lab[0]), str(uuid4())+'.png'))
    print('\n')

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    base_path = os.path.join(os.curdir, 'data', 'cifar100')

    os.mkdir(base_path)

    for name in range(100):
        os.mkdir(os.path.join(base_path, str(name)))
    
    save_pngs(x_train, y_train, base_path)
    save_pngs(x_test, y_test, base_path)