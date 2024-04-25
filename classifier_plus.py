'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify the remaining code:
1. Replace the random classifier with your trained model.(line 64-68)
2. modify the get_label function to get the predicted label.(line 18-24)(just like Leetcode solutions)
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import torch
import csv
import os
import pandas as pd
import numpy as np
from bidict import bidict
import csv
import json


NUM_CLASSES = len(my_bidict)

# Write your code here
# And get the predicted label, which is a tensor of shape (batch_size,)
# Begin of your code

def get_label(model, model_input, device):
    batch_size = model_input.size(0)
    all_predictions =  torch.zeros(NUM_CLASSES, batch_size, dtype=torch.float32, device=device)
    for i in range(NUM_CLASSES):
        # Convert label to tensor representation
        class_label = label_to_onehot_tensor([my_bidict.inverse[i]]*batch_size)

        # Forward pass through the model to get raw outputs
        raw_output = model(model_input, class_label=class_label)

        # Convert raw logistics into probabilities or logits
        # mean pooling over logistic parameters
        all_predictions[i] = discretized_mix_logistic_loss(model_input, raw_output, train=False)
        
    # Compute probabilities using softmax
    _, pred = torch.min(all_predictions, dim=0)
    
    return pred


def get_label_lg(model, model_input, device, logit_file):
    batch_size = model_input.size(0)
    all_predictions =  torch.zeros(NUM_CLASSES, batch_size, dtype=torch.float32, device=device)
    for i in range(NUM_CLASSES):
        # Convert label to tensor representation
        class_label = label_to_onehot_tensor([my_bidict.inverse[i]]*batch_size)
        class_label = torch.zeros(batch_size, NUM_CLASSES, device=device)
        class_label[:, i] = 1

        # Forward pass through the model to get raw outputs
        raw_output = model(model_input, class_label=class_label)

        # Convert raw logistics into probabilities or logits
        # mean pooling over logistic parameters
        all_predictions[i] = discretized_mix_logistic_loss(model_input, raw_output, train=False)
        
    # Compute probabilities using softmax
    _, pred = torch.min(all_predictions, dim=0)
    
    # write the logit to a npy file, file file exist, append to it
    if os.path.exists(logit_file):
        logits = np.load(logit_file)
        # Append new logits along the second dimension (i.e., for each class across batches)
        updated_logits = np.append(logits, all_predictions.detach().cpu().numpy(), axis=1)
        np.save(logit_file, updated_logits)
    else:
        # Save the logits for the first time.
        np.save(logit_file, all_predictions.detach().cpu().numpy())

    return pred

class CPEN455Dataset_path(Dataset):
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
        image_path = img_path
        return image, category_name, image_path

import torchvision.transforms as transforms

def record(model, data_loader, device, result_file_path, logit_file, mode="basic"):
    
    with open(result_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        
        for batch_idx, item in enumerate(tqdm(data_loader)):
            model_input, _, filename = item 
            # parth filename so its only the filename
            filenames = [os.path.basename(f) for f in filename]
            
            model_input = model_input.to(device)
            if mode == "smart":
                pass
            else:
                predictions = get_label_lg(model, model_input, device, logit_file).cpu().numpy()
            print(predictions)
            
            for filename, prediction in zip(filenames, predictions):
                writer.writerow([filename, prediction])
        
        writer.writerow(['fid', 455])

def record_wrong(paths, answer, original_label):
    """Save the wrong item to a file, it should have image file path as key, [correct, predicted] as value
    result should be json
    """
    if not os.path.exists('wrong_items.json'):
        file = open('wrong_items.json', 'w')
        file.write('{}')
        file.close()

    file = open('wrong_items.json', 'r')
    content = file.read()
    file.close()
    file = open('wrong_items.json', 'w')
    try:
        wrong_items = json.loads(content)
    except json.JSONDecodeError:
        wrong_items = {}
        print('Error in decoding json file')
    for i in range(len(answer)):
        if answer[i] != original_label[i]:
            # tensor to int
            wrong_items[paths[i]] = [original_label[i].item(), answer[i].item()]
    file.write(json.dumps(wrong_items))
    file.close()

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        print(answer)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()

def classifier_save(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories, path = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        print(answer)
        correct_num = torch.sum(answer == original_label)
        record_wrong(path, answer, original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()


# section debugged and fixed by GPT
def get_label_multi_region_smart(model, model_input, xy_set, device, zoom=False):
    batch_size = model_input.size(0)
    all_predictions = torch.zeros(NUM_CLASSES, batch_size, dtype=torch.float32, device=device)

    # Iterate over each region specified by the xy tuple
    for x, y in xy_set:
        # Masking the image to consider only pixels up to (x, y)
        mask = torch.zeros_like(model_input)
        mask[:, :, :x, :y+1] = 1  # Include all rows up to x, and all columns up to y
        # output channel is 1000, apply mask to every channel

        # create empty tensor with [modelinput.size(0), 1000, 32, 32]
        mask_out_t = torch.zeros((model_input.size(0), 1000, 32, 32), device=device)
        mask_out_t[:, :, :x, :y+1] = 1


        # Apply mask; portions of image beyond (x, y) are zeroed out
        masked_input = model_input * mask
        if zoom == True:
            # zoom into the image, ignore all mask area, resize to 32x32
            # input should have 3 channels
            masked_input = masked_input[:, :, :x, :y+1]
            masked_input = torch.nn.functional.interpolate(masked_input, size=(32, 32), mode='linear')
            
        region_predictions = torch.zeros(NUM_CLASSES, batch_size, dtype=torch.float32, device=device)
        for i in range(NUM_CLASSES):
            # Convert label to tensor representation
            class_label = label_to_onehot_tensor([my_bidict.inverse[i]]*batch_size)

            # Forward pass through the model to get raw outputs
            raw_output = model(masked_input, class_label=class_label, sample=True)

            # Evaluate loss only on the visible part of the image
            # To make sure the mask applies properly, we use the same mask on the output
            
            # apply mask, both should have same size
            masked_output = raw_output * mask_out_t

            # Calculate logistic loss for masked region
            region_predictions[i] = discretized_mix_logistic_loss(masked_input, masked_output, train=False)

        # Accumulate predictions across all specified regions by averaging logits
        all_predictions += region_predictions / len(xy_set)

    # Compute softmax probabilities to find classes and then find the minimum predicted class label
    _, pred = torch.min(all_predictions, dim=0)
    pred_2 = torch.argmin(torch.softmax(all_predictions, dim=0), dim=0)
    if not torch.equal(pred, pred_2):
        print('Error in prediction')
        print(pred)
        print(pred_2)

    return pred
    
def classifier_smart(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()  # Assuming 'ratio_tracker' is defined elsewhere to track accuracy
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = torch.tensor([my_bidict[item] for item in categories], dtype=torch.int64, device=device)
        
        xy_set = [(23, 32), (28, 32), (32, 32), (32, 28), (32, 23)]
        xy_set = [(32, 32)]
        answer = get_label_multi_region_smart(model, model_input, xy_set, device)
        correct_num = torch.sum(answer == original_label).item()
        acc_tracker.update(correct_num, model_input.shape[0])
        record_wrong(item, answer, original_label)
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=16, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #Write your code here
    #You should replace the random classifier with your trained model
    #Begin of your code
   
    fix_seeds()
    run_mode = "base"
    parms = {"nr_resnet": 1, "nr_filters": 128, "input_channels": 3, "nr_logistic_mix": 100}
    single_path = r"models/conditional_pixelcnn.pth"
    strict = False


    if run_mode == "all":
        
        model_path = r'.\models'
        eval_result = {}
        for file in os.listdir(model_path):
            if file.endswith('.pth'):
                model_path_full = os.path.join(model_path, file)
                model = PixelCNN(**parms)

                #End of your code

                model = model.to(device)
                #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
                #You should save your model to this path
                model.load_state_dict(torch.load(model_path_full), strict=strict)
                model.eval()
                print('model parameters loaded')
                acc = classifier(model = model, data_loader = dataloader, device = device)
                print(f"Accuracy: {acc}")
                eval_result[file] = acc
                # clean
                del model
                torch.cuda.empty_cache()
        # print from highest acc to lowest
        eval_result = dict(sorted(eval_result.items(), key=lambda item: item[1], reverse = True))
        print(eval_result)
        # print best 5 model as list format
        print(list(eval_result.items())[:5])

    elif run_mode == "test":
        base = 'models/'
        model_list = [
        'conditional_pixelcnn.pth',
        'pcnn_cpen455_load_model_134.pth',
        'pcnn_cpen455_load_model_149.pth'
        ]
        dataloader_t = torch.utils.data.DataLoader(CPEN455Dataset_path(root_dir=args.data_dir, 
                                                            mode = "test", 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)
        
        logit_base = "logits/"
            
        #End of your code
        fix_seeds()
        
        for model_path_full in model_list:
            logit_file = model_path_full.replace('.pth', '_logits.npy')
            logit_file = os.path.join(logit_base, logit_file)
            if os.path.exists(logit_file):
                os.remove(logit_file)

            model = PixelCNN(**parms)

            #End of your code

            model = model.to(device)
            #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
            #You should save your model to this path
            model.load_state_dict(torch.load(base + model_path_full), strict=strict)
            model.eval()
            model_name = os.path.basename(model_path_full)
            # strip pcnn_cpen455_from_, if it exist
            model_name = model_name.replace('pcnn_cpen455_from_', '')
            record(model, dataloader_t, device, 'results_' + model_name + '.csv', logit_file=logit_file)
            # clean
            del model
            torch.cuda.empty_cache()

    elif run_mode == "base":
        dataloader_t = torch.utils.data.DataLoader(CPEN455Dataset_path(root_dir=args.data_dir, 
                                                            mode = "validation", 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)
        model = PixelCNN(**parms)
        model = model.to(device)
        #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
        #You should save your model to this path
        model.load_state_dict(torch.load(single_path), strict=strict)
        model.eval()
        print('model parameters loaded')
        
        acc = classifier_save(model = model, data_loader = dataloader_t, device = device)
        print(f"Accuracy: {acc}")

    elif run_mode == "smart":
        dataloader_t = torch.utils.data.DataLoader(CPEN455Dataset_path(root_dir=args.data_dir, 
                                                            mode = "validation", 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)
        model = PixelCNN(**parms)
        model = model.to(device)
        #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
        #You should save your model to this path
        model.load_state_dict(torch.load('models/conditional_pixelcnn.pth'))
        model.eval()
        print('model parameters loaded')
        
        acc = classifier_smart(model = model, data_loader = dataloader_t, device = device)
        print(f"Accuracy: {acc}")
        
        