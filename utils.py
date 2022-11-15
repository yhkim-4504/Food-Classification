import random
import cv2
import mlflow
from os.path import isdir, basename, join
from glob import glob


def split_train_valid(dataset_path, train_ratio, rnd_state):
    label_to_idx = {basename(p): i for i, p in enumerate(sorted(glob(join(dataset_path, '*')))) if isdir(p)}
    
    random.seed(rnd_state)
    img_paths = [x for x in sorted(glob(join(dataset_path, '*/*.*g')))]
    
    trainset_paths = random.sample(img_paths, round(len(img_paths)*train_ratio))
    validset_paths = [p for p in img_paths if p not in trainset_paths]
    
    return label_to_idx, trainset_paths, validset_paths

def load_config_to_mlflow(config, **kwargs):
    for key, value in config.items():
        print('------------------------------')
        if key in kwargs:
            print(f'{key}: {kwargs[key]}')
            mlflow.log_param(key, kwargs[key])
            if config[key][kwargs[key]]:
                print(config[key][kwargs[key]])
                mlflow.log_params(config[key][kwargs[key]])
        else:
            print(key)
            print(value)
            if value:
                mlflow.log_params(value)
    print()

def pad_to_square(img):
    if img.shape[0]==img.shape[1]:
        return img
    
    length = max(img.shape)
    delta_w = length - img.shape[1]
    delta_h = length - img.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    pad_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 255])
    
    return pad_img