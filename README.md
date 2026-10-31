# mnist-neural-network-from-scratch
MNIST digit classification using Numpy and TensorFlow
# MNIST Handwritten Digit Classification
### Neural Network from Scratch vs. TensorFlow/Keras

## Overview

This project explores the mathematical foundations and practical implementation of neural networks through handwritten digit classification using the MNIST dataset.

I developed two neural network implementations:

1. **From Scratch (NumPy):** Implemented the core neural network algorithms manually without using a deep learning framework.
2. **TensorFlow/Keras:** Developed a second model using an established deep learning framework to compare performance and implementation approaches.

The objective was not only to achieve high classification accuracy but also to understand the underlying mechanisms of neural network training.

## Implementation

### 1. Neural Network from Scratch

Built a neural network using Python and NumPy, manually implementing:

- Forward propagation
- Backpropagation
- Gradient descent
- Model training and evaluation

This implementation helped me explore how neural networks learn from data and how mathematical operations are translated into code.

### 2. TensorFlow/Keras Implementation

Developed a second neural network using TensorFlow/Keras to investigate how deep learning frameworks simplify model development and training.

Compared the two implementations in terms of classification accuracy, loss, and execution time.

## Results

| Metric | NumPy (From Scratch) | TensorFlow/Keras |
|---|---:|---:|
| Accuracy | **98.10%** | 97.86% |
| Loss | 0.0744 | 0.0747 |
| Execution Time | 30.2 s | 19.6 s |

The custom NumPy implementation achieved slightly higher classification accuracy than the TensorFlow/Keras implementation in this experiment, while the TensorFlow/Keras model completed execution faster.

These results illustrate the trade-offs between manually implementing neural network algorithms and using optimized machine learning frameworks.

## Technologies Used

- Python
- NumPy
- TensorFlow/Keras
- MNIST Dataset

## Key Learnings

Through this project, I developed a deeper understanding of:

- The mathematical operations underlying neural networks.
- How forward propagation and backpropagation enable learning.
- The role of gradient descent in optimizing model parameters.
- How machine learning frameworks abstract complex mathematical operations.
- How to evaluate and compare machine learning implementations using quantitative metrics.

## Conclusion

This project strengthened my understanding of neural network fundamentals by connecting mathematical concepts with practical programming. Implementing a neural network from scratch provided insight into the algorithms underlying modern deep learning frameworks, while the TensorFlow/Keras comparison helped me evaluate different approaches to model development.
