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
from classification_evaluation import fix_seeds

NUM_CLASSES = len(my_bidict)

# Write your code here
# And get the predicted label, which is a tensor of shape (batch_size,)
# Begin of your code
def get_label_single(model, model_input, device):
    batch_size = model_input.size(0)
    all_predictions = torch.zeros(NUM_CLASSES, batch_size, dtype=torch.float32, device=device)
    
    for j in range(batch_size):
        single_input = model_input[j].unsqueeze(0)  # Get a single image and maintain it as a batch of size 1
        # print size
        print(single_input.size())
        for i in range(NUM_CLASSES):
            # Convert label to tensor representation for a single example
            class_label = torch.zeros(1, NUM_CLASSES, device=device)  # Adjusted for batch size of 1
            class_label[:, i] = 1

            # Forward pass through the model to get raw outputs for a single input
            raw_output = model(single_input, class_label=class_label)

            # Convert raw logistics into probabilities or logits
            # mean pooling over logistic parameters for a single input
            all_predictions[i, j] = discretized_mix_logistic_loss(single_input, raw_output, train=False)
    
    # Compute probabilities using softmax, and find predictions
    _, pred = torch.min(all_predictions, dim=0)  # No change required here

    # Additional check for prediction correctness
    pred_2 = torch.argmin(torch.softmax(all_predictions, dim=0), dim=0)

    logit_file = 'logits.npy'
    if os.path.exists(logit_file):
        logits = np.load(logit_file)
        # Append new logits along the second dimension (i.e., for each class across batches)
        updated_logits = np.append(logits, all_predictions.detach().cpu().numpy(), axis=1)
        np.save(logit_file, updated_logits)
    else:
        # Save the logits for the first time.
        np.save(logit_file, all_predictions.detach().cpu().numpy())

    print('Error in prediction')
    print(pred)
    print(pred_2)
    
    return pred

def get_label(model, model_input, device):
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
    pred_2 = torch.argmin(torch.softmax(all_predictions, dim=0), dim=0)
    # write the logit to a npy file, file file exist, append to it
    logit_file = 'logits.npy'
    if os.path.exists(logit_file):
        logits = np.load(logit_file)
        # Append new logits along the second dimension (i.e., for each class across batches)
        updated_logits = np.append(logits, all_predictions.detach().cpu().numpy(), axis=1)
        np.save(logit_file, updated_logits)
    else:
        # Save the logits for the first time.
        np.save(logit_file, all_predictions.detach().cpu().numpy())

    if not torch.equal(pred, pred_2):
        print('Error in prediction')
        print(pred)
        print(pred_2)
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
def record(model, data_loader, device, result_file_path):
    
    with open(result_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        
        for batch_idx, item in enumerate(tqdm(data_loader)):
            model_input, _, filename = item 
            # parth filename so its only the filename
            filenames = [os.path.basename(f) for f in filename]
            
            model_input = model_input.to(device)
            predictions = get_label(model, model_input, device).cpu().numpy()
            print(predictions)
            
            for filename, prediction in zip(filenames, predictions):
                writer.writerow([filename, prediction])
        
        writer.writerow(['fid', 455])

# End of your code

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=1, help='Batch size for inference')
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
    model = PixelCNN(nr_resnet=2, nr_filters=160, input_channels=3, nr_logistic_mix=100)
    dataloader_t = torch.utils.data.DataLoader(CPEN455Dataset_path(root_dir=args.data_dir, 
                                                            mode = "test", 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)
    # clear logit file:
    logit_file = 'logits.npy'
    if os.path.exists(logit_file):
        os.remove(logit_file)

    #End of your code
    fix_seeds()
    base = 'models/'
    model_list = [
    'pcnn_cpen455_from_scratch_499.pth',
    'pcnn_cpen455_from_scratch_474.pth',
    'pcnn_cpen455_from_scratch_324.pth',
    'pcnn_cpen455_from_scratch_399.pth',
    'pcnn_cpen455_from_scratch_424.pth'
]
    for model_path_full in model_list:

        model = PixelCNN(nr_resnet=1, nr_filters=128, input_channels=3, nr_logistic_mix=100)

        #End of your code

        model = model.to(device)
        #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
        #You should save your model to this path
        model.load_state_dict(torch.load(base + model_path_full))
        model.eval()
        model_name = os.path.basename(model_path_full)
        # strip pcnn_cpen455_from_, if it exist
        model_name = model_name.replace('pcnn_cpen455_from_', '')
        record(model, dataloader_t, device, 'results_' + model_name + '.csv')
        # clean
        del model
        torch.cuda.empty_cache()
  
    
    print('model parameters loaded')
        
        