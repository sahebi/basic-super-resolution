import tarfile
from os import remove
from os.path import exists, join, basename

from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

import sys
sys.path.append('./')
from . import dataset

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

DATASET_FILE_NAME = './dataset/dataset.yml'

def download_dataset(dest="./", data='BSDS300'):
    url = data['url']
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

def get_data(dataset_name, data_type='test', upscale_factor=2):
    with open(DATASET_FILE_NAME, 'r') as stream:
        try:
            dataset_yml = yaml.safe_load(stream)
            data_dir    = dataset_yml[dataset_name]['path'] + dataset_yml[dataset_name][data_type]
        except yaml.YAMLError as exc:
            print(exc)
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return dataset.DatasetFromFolder(data_dir,
                            input_transform=input_transform(crop_size, upscale_factor),
                            target_transform=target_transform(crop_size))

def get_training_set(dataset_name='COCO', upscale_factor=2):
    with open(DATASET_FILE_NAME, 'r') as stream:
        try:
            dataset_yml = yaml.safe_load(stream)
            train_dir   = dataset_yml[dataset_name]['path'] + dataset_yml[dataset_name]['train']
        except yaml.YAMLError as exc:
            print(exc)
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return dataset.DatasetFromFolder(train_dir,
                            input_transform=input_transform(crop_size, upscale_factor),
                            target_transform=target_transform(crop_size))

def get_test_set(dataset_name = 'COCO', upscale_factor=2):
    with open(DATASET_FILE_NAME, 'r') as stream:
        try:
            dataset_yml = yaml.safe_load(stream)
            test_dir = dataset_yml[dataset_name]['path'] + dataset_yml[dataset_name]['test']
        except yaml.YAMLError as exc:
            print(exc)
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return dataset.DatasetFromFolder(test_dir,
                            input_transform=input_transform(crop_size, upscale_factor),
                            target_transform=target_transform(crop_size))
