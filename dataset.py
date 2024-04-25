import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize
from bidict import bidict
from tqdm import tqdm
import pandas as pd
import pdb
import random

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
replicate_color_channel = lambda x : x.repeat(3,1,1)

my_bidict = bidict({'Class0': 0, 
                    'Class1': 1,
                    'Class2': 2,
                    'Class3': 3})

class CPEN455Dataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        ROOT_DIR = './data'
        root_dir = os.path.join(root_dir, mode)
        self.root_dir = root_dir
        self.transform = transform
        """self.samples have structure like this
        is a list of tuples, each tuple is a pair of image path and its label
        """
        self.samples = []  # List to store image paths along with domain and category
        # Walk through the directory structure
        csv_path = os.path.join(ROOT_DIR, mode + '.csv')
        df = pd.read_csv(csv_path, header=None, names=['path', 'label'])
        # Convert DataFrame to a list of tuples
        self.samples = list(df.itertuples(index=False, name=None))
        self.samples = [(os.path.join(ROOT_DIR, path), label) for path, label in self.samples]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, category = self.samples[idx]
        if category in my_bidict.values():
            category_name = my_bidict.inverse[category]
        else:
            category_name = "Unknown"
        # print(img_path)
        image = read_image(img_path)  # Reads the image as a tensor
        image = image.type(torch.float32) / 255.  # Normalize to [0, 1]
        if image.shape[0] == 1:
            image = replicate_color_channel(image)
        if self.transform:
          image = self.transform(image)
        return image, category_name
    
    def get_all_images(self, label):
        return [img for img, cat in self.samples if cat == label]
    
class CPEN455Dataset_ft(Dataset):
    def __init__(self, root_dir, classes:list, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            classes: list of classes sublist of [0, 1, 2, 3], obtained image should 90% come for the specified classes
        """
        ROOT_DIR = './data'
        root_dir = os.path.join(root_dir, "train")
        root_dir2= os.path.join(root_dir, "validation")
        self.root_dir = root_dir
        self.root_dir_2 = root_dir2
        self.transform = transform
        """self.samples have structure like this
        is a list of tuples, each tuple is a pair of image path and its label
        """
        self.samples = []  # List to store image paths along with domain and category
        # Walk through the directory structure
        # Divide load in the ratio of 30% from training and 70% from validation
        train_frac, val_frac = 0.3, 0.7

        # gpt generated
        csv_path = os.path.join(ROOT_DIR, "train" + '.csv')
        df = pd.read_csv(csv_path, header=None, names=['path', 'label'])

        csv_path2 = os.path.join(ROOT_DIR, "validation" + '.csv')
        df2 = pd.read_csv(csv_path2, header=None, names=['path', 'label'])

        desired_classes = []  #Fill this with the two classes which your model confuses
        rest_classes = []     #Fill this with rest of the classes

        # Filter desired classes and rest classes from df
        subset_df = df[df['label'].isin(desired_classes)]
        rest_df = df[df['label'].isin(rest_classes)]

        desired_val_df = df2[df2['label'].isin(desired_classes)]
        rest_val_df = df2[df2['label'].isin(rest_classes)]

        # Get 90% of total from desired classes and 10% from rest
        desired_num = round(train_frac * 0.9 * (min(len(subset_df), len(desired_val_df))))
        rest_num = round(train_frac * 0.1 * (min(len(rest_df), len(rest_val_df))))

        samples_train = random.sample(list(subset_df.itertuples(index=False, name=None)), desired_num)
        samples_train.extend(random.sample(list(rest_df.itertuples(index=False, name=None)), rest_num))

        # Repeat the process for validation data
        desired_num_val = round(val_frac * 0.9 * (min(len(subset_df), len(desired_val_df))))
        rest_num_val = round(val_frac * 0.1 * (min(len(rest_df), len(rest_val_df))))

        samples_val = random.sample(list(desired_val_df.itertuples(index=False, name=None)), desired_num_val)
        samples_val.extend(random.sample(list(rest_val_df.itertuples(index=False, name=None)), rest_num_val))

        self.samples = [(os.path.join(ROOT_DIR, path), label) for path, label in samples_train]
        self.samples2 = [(os.path.join(ROOT_DIR, path), label) for path, label in samples_val]
        
    def __len__(self):
        return len(self.samples + self.samples2)

    def __getitem__(self, idx):
        img_path, category = self.samples[idx]
        if category in my_bidict.values():
            category_name = my_bidict.inverse[category]
        else:
            category_name = "Unknown"
        # print(img_path)
        image = read_image(img_path)  # Reads the image as a tensor
        image = image.type(torch.float32) / 255.  # Normalize to [0, 1]
        if image.shape[0] == 1:
            image = replicate_color_channel(image)
        if self.transform:
          image = self.transform(image)
        return image, category_name
    
    def get_all_images(self, label):
        return [img for img, cat in self.samples if cat == label]

def show_images(images, categories, mode:str):
        fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
        for i, image in enumerate(images):
            axs[i].imshow(image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            axs[i].set_title(f"Category: {categories[i]}")
            axs[i].axis('off')
        plt.savefig(mode + '_test.png')

if __name__ == '__main__':
    
    transform_32 = Compose([
        Resize((32, 32)),  # Resize images to 32 * 32
        rescaling
    ])
    dataset_list = ['train', 'validation', 'test']
    
    for mode in dataset_list:
        print(f"Mode: {mode}")
        dataset = CPEN455Dataset(root_dir='./data', transform=transform_32, mode=mode)
        data_loader = DataLoader(dataset, batch_size = 4, shuffle=True)
        # Sample from the DataLoader
        for images, categories in tqdm(data_loader):
            print(images.shape, categories)
            images = torch.round(rescaling_inv(images) * 255).type(torch.uint8)
            show_images(images, categories, mode)
            break  # We only want to see one batch of 4 images in this example
        