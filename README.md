# MNIST Digit Classifier in C++

A neural network implementation for classifying handwritten digits from the MNIST dataset using C++. The project implements a simple feedforward neural network with one hidden layer using ReLU activation, following the architecture described in Chapter 8 "Learning signal and ignoring noise" of the book "Grokking Deep Learning" by Andrew W. Trask.

## Features

- Custom matrix operations implementation
- Neural network with one hidden layer
- ReLU activation function
- MNIST dataset handling

## Prerequisites

- C++ compiler with C++11 support
- Make build system
- MNIST dataset files:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`

## Project Structure

```
mnist-classifier-cpp/
│
├── mlmath.h         - Matrix operations and math
├── mnist.h          - MNIST dataset definitions
├── main.cpp         - Neural network implementation
└── Makefile         - Build configuration
```

## Building and Running

1. Place the MNIST dataset files in a `dataset` folder in the project root
2. Build the project:

```bash
make main.cpp
```

3. Run the classifier:

```bash
./main.exe
```

## Neural Network Architecture

The neural network follows the implementation from "Grokking Deep Learning" Chapter 8:

- Input layer: 784 neurons (28x28 pixels)
- Hidden layer: 40 neurons with ReLU activation
- Output layer: 10 neurons (one for each digit)

## Training Parameters

- Learning rate (alpha): 0.005
- Epochs: 50
- Training set size: 1000 samples
- Hidden layer size: 40 neurons
