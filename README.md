# neural-network-image-classifier

## Table of contents:
- [Project overview](#project-overview)
- [Architechture](#architechture)
- [Training details](#training-details)
- [Results](#results)
- [How to use](#how-to-use)

## Project overview:

This personal project outlines the development and evaluation of a convolutional neural network (CNN) for CIFAR-10 image classification. Through careful hyperparameter tuning, regularisation, and data augmentation, the model achieved a test accuracy of 89.13% from 30%.

## Architechture: 
The basic architecture consisted of custom "IntermediateBlock" modules. Each block processed an image through multiple convolutional paths (e.g., 3x3 convolutions) and used attention weights generated from the average pooled channel data to combine the results. The output block consisted of a global average pooling layer followed by a fully connected linear layer to produce a 10-class logits output.
A simple CNN was used to test the training pipeline, achieving a baseline test accuracy of approximately 30–35%.

To improve accuracy, the final model was expanded to include six IntermediateBlocks, each containing three convolutional paths with kernel sizes of 1, 3, and 5. Attention was applied over these paths using a softmax layer. Key improvements included:
• Parallel convolutions in each block (1x1, 3x3, 5x5) for multi-scale feature extraction
• Attention mechanism for adaptive weighting of convolution outputs
• Residual connections to improve gradient flow
• Batch Normalisation and Dropout (p=0.3) to regularise training
• OutputBlock with a hidden layer of 512 units and ReLU activation

## Training details:
I improved the test accuracy of my neural network by  selecting a range of hyperparameters. I used the AdamW optimiser with a learning rate of 0.01, which allowed the
model to learn effectively in the early stages of training. To regularise the model and prevent overfitting, I applied weight decay set to 1e-4 and added dropout with a rate of 0.3 both after the convolutional layers in each intermediate block and in the output block. The model was trained for 120 epochs, using a batch size of 128. This batch size provided stable
gradient estimates and made efficient use of the GPU. I also used a cosine annealing learning rate scheduler with T_max set to 200 to gradually reduce the learning rate during training, which helped the model settle into a good solution and improve accuracy over time. The loss function used was CrossEntropyLoss, which is appropriate for multi-class classification tasks like CIFAR-10.

The final architecture consisted of six IntermediateBlocks, each containing three parallel convolutional paths with kernel sizes of 1×1, 3×3, and 5×5. These paths enabled
the model to capture multi-scale features from the input. Each block also used batch normalisation, ReLU activations, and residual (shortcut) connections to support deeper learning
and better gradient flow. The attention mechanism computed soft weights from the average pooled input to combine the outputs of the three convolutional paths adaptively.
The output classifier used 512 hidden units with dropout, followed by a linear layer to produce final logits. Data augmentation techniques included random cropping, flipping, colour jitter, and rotation to help the model generalise better.

## Results:
Highest Test Accuracy: 89.13% (achieved at epoch 108)

Image 1 shows : Test Accuracy per Epoch (final model) - The test accuracy steadily improved during training and peaked at 89.13% (epoch 108). Small fluctuations near the peak are normal and show the model generalised well.

<img width="849" height="393" alt="Unknown-1" src="https://github.com/user-attachments/assets/d18db7f5-2353-49c8-9541-6e4c2cc457e6" />

Image 2 shows : Training vs. Test Accuracy Comparison (final model) - Both curves followed a similar upward trend with only a small gap, indicating minimal overfitting. Regularisation methods such as dropout, weight decay, and data augmentation successfully kept the model balanced.

<img width="854" height="393" alt="Unknown" src="https://github.com/user-attachments/assets/b747afe6-d85b-4813-b07f-34fb26098687" />

## How to use:
- Open the Jupyter Notebook file (`Neural_networks_cw1.ipynb`) to see the full implementation. 
- Results, including accuracy curves and final performance, are provided in the Results section.  



