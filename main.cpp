#include <iostream>
#include "mnist.h"
#include "mlmath.h"
#include <math.h>

// define function that convert images to vector of mlm::Matrix
std::vector<mlmath::Matrix> imagesToMatrix(const mnist::MNISTImages &images)
{
    std::vector<mlmath::Matrix> result;
    for (int i = 0; i < images.numImages; i++)
    {
        mlmath::Matrix matrix(images.numRows, images.numCols);
        for (int j = 0; j < images.numRows; j++)
        {
            for (int k = 0; k < images.numCols; k++)
            {
                matrix.data[j][k] = images.images[i][j * images.numCols + k];
            }
        }

        result.push_back(matrix);
    }
    return result;
}

// one-hot for labels
std::vector<mlmath::Matrix> oneHot(const mnist::MNISTLabels &labels)
{
    std::vector<mlmath::Matrix> result;
    for (int i = 0; i < labels.numLabels; i++)
    {
        mlmath::Matrix matrix(10, 1);
        for (int j = 0; j < 10; j++)
        {
            matrix.data[j][0] = labels.labels[i] == j ? 1 : 0;
        }
        result.push_back(matrix);
    }
    return result;
}

int main()
{

    const std::string trainImagesPath = "dataset/train-images.idx3-ubyte";
    const std::string trainLabelsPath = "dataset/train-labels.idx1-ubyte";

    mnist::MNISTImages rowImages(trainImagesPath);
    mnist::MNISTLabels rowLabels(trainLabelsPath);

    std::vector<mlmath::Matrix> images = imagesToMatrix(rowImages);
    std::vector<mlmath::Matrix> labels = oneHot(rowLabels);

    const double alpha = 0.005;
    const int epochs = 50;
    const int hiddenLayerSize = 40;
    const int pixelsPerImage = rowImages.numRows * rowImages.numCols;
    const int numLabels = 10;
    const int trainTestSize = 1000;

    std::cout << "Check training args: " << std::endl;
    std::cout << "Alpha: " << alpha << " Epochs: " << epochs << " Hidden Layer Size: " << hiddenLayerSize << " Pixels Per Image: " << pixelsPerImage << " Num Labels: " << numLabels << std::endl;

    mlmath::Matrix weights_0_1 = mlmath::Matrix::random(pixelsPerImage, hiddenLayerSize, -0.1, 0.1); // Shape (784, 40)
    mlmath::Matrix weights_1_2 = mlmath::Matrix::random(hiddenLayerSize, numLabels, -0.1, 0.1);      // Shape (40, 10)

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double error = 0.0;
        int correct_count = 0;

        for (int i = 0; i < trainTestSize; i++)
        {
            // Forward pass
            mlmath::Matrix layer_0 = images[i].reshape(1, pixelsPerImage) / 255.0; // Shape (1, 784)
            mlmath::Matrix layer_1 = layer_0 * weights_0_1;                        // Shape (1, 40)
            layer_1 = mlmath::relu(layer_1);
            mlmath::Matrix layer_2 = layer_1 * weights_1_2; // Shape (1, 10)

            // Error calculation
            mlmath::Matrix target = labels[i].transpose(); // Shape (1, 10)
            error += ((target - layer_2) ^ 2.0).sum();
            correct_count += mlmath::argmax(layer_2) == mlmath::argmax(labels[i]);

            // Backpropagation
            mlmath::Matrix layer_2_delta = layer_2 - target;                         // Shape (1, 10)
            mlmath::Matrix layer_1_delta = (layer_2_delta * weights_1_2.transpose()) // Shape (1, 40)
                                               .elementWiseMultiply(mlmath::relu_derivative(layer_1));

            // Weight updates
            weights_1_2 -= (layer_1.transpose() * layer_2_delta) * alpha; // Shape (40, 10)
            weights_0_1 -= (layer_0.transpose() * layer_1_delta) * alpha; // Shape (784, 40)
        }

        // print the number of epoch with error and accuracy divided by trainTestSize
        std::cout << "Epoch: " << epoch << " Error: " << error / trainTestSize << " Accuracy: " << (double)correct_count / trainTestSize << std::endl;
    }
    return 0;
}
