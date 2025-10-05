# Conditional CNN for Image Classification

PixelCNNG is an attempt to convert an unlabeled generative model [PixelCNN:https://arxiv.org/abs/1606.05328] into a predictive model by adding a label channel and forcing the final generation probability to map to classes.

Topics Explored:
1. Single-class generative model -> Class-based generative model
2. Using weights from a generative model in a classification task by adding a new classification layer
3. Study the impact of input transformations (rotation, flip, filter) on model performance


