# CPEN455 2023 W2 Course Project: Conditional PixelCNN++ for Image Classification

Author: Guan Zheng Huang
ID: 67321109

## Statement on resource use:

- Single GPU of 2070 Super Max-Q Laptop Processor
- Time to train main mode: 16 Hours + 2 Hours Finetune.
- Time to train reduced parameter model of [resnet = 1, filter = 40, logistical mix = 10] to epoch 350 took 3 hours.
- The code consist of only packages listed under requirements.txt. 
- The training process took approximately 24 hours to complete. 
- Cuda version is 12.4.131, 
- Torch version is 2.2.1 cuda121
- Python version is 3.10.13.
- All seed and randomness is locked, including inference, training and finetuning process. Reference code for seed used.

## Statement on Overfitting to test
- Inevitably, a reasonably successful model on the validation dataset are ran against te test set, meaning that the final model may overfit to the test set. 
- To mitigate this issue, I ensured that each model with unique set of parameters only ran against the test set for a maximum of 3 times among the best 3 performing checkpoint on validation set.

## Statement on code cleanses
- The code is clean and most code that interfeers with the understanding of the 
