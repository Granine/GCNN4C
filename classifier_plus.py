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
    pred_2 = torch.argmin(torch.softmax(all_predictions, dim=0), dim=0)
    if not torch.equal(pred, pred_2):
        print('Error in prediction')
        print(pred)
        print(pred_2)
    return pred

import random
 # Set seeds for reproducibility
def fix_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# End of your code

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
    
    model_path = r'.\models'
    eval_result = {}
    for file in os.listdir(model_path):
        if file.endswith('.pth'):
            model_path_full = os.path.join(model_path, file)
            model = PixelCNN(nr_resnet=1, nr_filters=128, input_channels=3, nr_logistic_mix=100)

            #End of your code

            model = model.to(device)
            #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
            #You should save your model to this path
            model.load_state_dict(torch.load(model_path_full))
            model.eval()
            print('model parameters loaded')
            acc = classifier(model = model, data_loader = dataloader, device = device)
            print(f"Accuracy: {acc}")
            eval_result[file] = acc
            # clean
            del model
            torch.cuda.empty_cache()
            
    print(eval_result)
        
        