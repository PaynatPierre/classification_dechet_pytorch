from torch.utils.data import Dataset, DataLoader
from config import cfg
import os
import random
import matplotlib.pyplot as plt
from matplotlib import image
import torch
import numpy as np
import torchvision


class image_dataset(Dataset):
    '''
    this class is made to load your data, preprocess them and give them to your dataloader

    data_paths: list which contains the path of all data you want to manage with this dataset
    train: Boolean representing if your dataset is a training dataset or not

    return: None
    '''
    def __init__(self, data_paths, train):
        self.data_paths = data_paths

        self.labels=[]
        for i in range(len(self.data_paths)):
            tmp_label = self.data_paths[i].split('/')[-2]
            self.labels.append(cfg.DATASET.LABELS.index(tmp_label))

        self.vertical_random_flip = torchvision.transforms.RandomVerticalFlip(p=0.5)
        self.horizontal_random_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)

        self.train_mode = train
        self.reshaper = torchvision.transforms.Resize((cfg.TRAIN.IMAGE_SHAPE))

    '''
    this function is made to give you the data at the index idx

    idx: integer representing the index of the data you want to get

    return:
        x_data: a tensor which contains your data
        labels: a tensor which contains the corresponding labels
    '''
    def __getitem__(self, idx):
        x_data = torch.tensor(image.imread(self.data_paths[idx]))
        x_data = torch.swapaxes(x_data, -1, 0)
        x_data = torch.swapaxes(x_data, -1, 1)
        x_data = self.reshaper(x_data)

        if self.train_mode:
            x_data = self.augmentation(x_data)

        # return x_data.float(), torch.nn.functional.one_hot(torch.tensor(self.labels[idx]), len(cfg.DATASET.LABELS)).float()
        return x_data.float(), torch.tensor(self.labels[idx])
    
    '''
    this function is made to apply data augmentation on your data

    args:
        data: tensor which contains your data
    
    return:
        data: tensor which contains your data modified

    '''
    def augmentation(self, data):
        data = self.vertical_random_flip(data)
        data = self.horizontal_random_flip(data)

        nbr_rotation = np.random.randint(0,2)*2
        data = torch.rot90(data, nbr_rotation, [1,2])

        return data
    
    '''
    this function is made to return your dataset's len

    args:
        None
    
    return: integer which represent the len of your dataset
    '''
    def __len__(self):
        return len(self.data_paths)
    

'''
this function is made to build train, validation and test dataloader

args:
    none

return:
    train_dataloader: instance of dataloader which deal with your training data
    val_dataloader: instance of dataloader which deal with your validation data
    test_dataloader: instance of dataloader which deal with your test data
'''
def get_dataloader():

    data_paths = []
    for folder in os.listdir(cfg.DATASET.DATA_FOLDER_PATH):
        for image in os.listdir(os.path.join(cfg.DATASET.DATA_FOLDER_PATH, folder)):
            data_paths.append(os.path.join(os.path.join(cfg.DATASET.DATA_FOLDER_PATH, folder),image))

    random.Random(4).shuffle(data_paths)

    train_dataloader = DataLoader(image_dataset(data_paths[:int(cfg.DATASET.TRAIN_PROPORTION*len(data_paths))], True), batch_size=cfg.DATASET.TRAINING_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(image_dataset(data_paths[int(cfg.DATASET.TRAIN_PROPORTION*len(data_paths)):int((cfg.DATASET.TRAIN_PROPORTION+cfg.DATASET.VALIDATION_PROPORTION)*len(data_paths))], False), batch_size=cfg.DATASET.VALIDATION_BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(image_dataset(data_paths[int((cfg.DATASET.TRAIN_PROPORTION+cfg.DATASET.VALIDATION_PROPORTION)*len(data_paths)):], False), batch_size=cfg.DATASET.TEST_BATCH_SIZE, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader

'''
this function is made to visualise you batch of data and your batch of label

args:
    None

return:
    None
'''
def show_batch():
    train_dataloader, val_dataloader, test_dataloader = get_dataloader()

    for i, (datas, labels) in enumerate(train_dataloader):
        for i in range(len(datas)):
            img = torch.swapaxes(datas[i], -1, 0)
            img = torch.swapaxes(img, 1, 0).type(torch.uint8)

            img = img.numpy()

            image.imsave(f'./tmp/img_{i}.png',img)
        break
