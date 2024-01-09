# Fashion-MNIST-Classifier-Project

## Project Overview

In this project, I tackled the challenge of classifying images from the Fashion-MNIST dataset, a collection of Zalando's article images. The dataset includes 60,000 training examples and 10,000 test examples, each being a 28x28 grayscale image categorized into one of 10 classes. Fashion-MNIST serves as an alternative to the original MNIST dataset, providing a more challenging classification task. The aim was to use Machine Learning, specifically through PyTorch, to build a Multi-Layered Perceptron (MLP) system that efficiently classifies these fashion items with high accuracy.

## Dataset Description

Fashion-MNIST is a dataset of Zalando's article images, each being a 28x28 grayscale image. It's divided into a training set of 60,000 examples and a test set of 10,000 examples, with each example associated with one of 10 classes. This dataset is a drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms.

## 2.1 Building a Vanilla Classifier

2.1.1 Dependencies
Python: 3.6
PyTorch: 1.3.1
Tensorboard: 1.14.0
Numpy: 1.16.6
termcolor: 1.1.0


2.1.2 Starting with the Data
Using PyTorch, I handled the large dataset with the torch.utils.data.dataloader class. This class facilitates batch processing and parallelization. I also created a custom “Dataset” class by inheriting torch.utils.data.Dataset and overriding __len__ and __getitem__ methods to handle our specialized dataset.

Purpose of __len__ and __getitem__:
__len__ returns the size of the dataset.
__getitem__ accesses individual data points.

## 2.1.3 Designing the Architecture
The architecture of the MLP includes one hidden layer with a structure of 784 → 100 → 10. I used ReLU activation after the first layer and, since PyTorch's cross-entropy loss criterion includes a Softmax layer, I did not apply Softmax explicitly.

## 2.1.4 A Holistic View of the Training
I utilized TensorBoard for visualizing various metrics during training:

Training Loss
Test Accuracy
Current learning rate

## 2.1.5 Training the Model
Training involved several steps:
Creating a cross-entropy loss criterion.
Using Stochastic Gradient Descent as the optimizer.
Implementing backpropagation.
Applying ReduceLROnPlateau for learning rate scheduling.
Running the model over all data points.
Aiming for at least 75% accuracy on the test set within 20 epochs.

## 2.2 Improving our Vanilla Classifier

### Weight Initialization
I experimented with two weight initialization strategies:
 - All weights and biases as zeros.
 - Xavier Normal initialization for weights and zeros for biases.
 - 
### Data Augmentation
After discussing the utility of data augmentation in general, I applied specific augmentation techniques to the dataset, analyzing the impact on the classifier's performance.

### Dropout Implementation
I incorporated dropout into the network, assessing its effect on the performance and stability of the model.

## TL;DR for Reporting Results

To ensure all aspects were covered, the final report included:

Purpose of __len__ and __getitem__ in the dataloader class.
Role of patience and factor parameters in the learning rate scheduler.
Tensorboard screenshots, final train loss, and test accuracy for various configurations of the classifier.
This project not only enhanced my understanding of neural network architectures and PyTorch but also provided practical experience in handling and analyzing image data. The challenges of tuning and improving the model gave me valuable insights into the intricacies of machine learning workflows.
