#include <iostream>
#include "mnist.h"

int main()
{
    const std::string trainImagesPath = "dataset/train-images.idx3-ubyte";
    const std::string trainLabelsPath = "dataset/train-labels.idx1-ubyte";

    mnist::MNISTImages images(trainImagesPath);
    mnist::MNISTLabels labels(trainLabelsPath);

    std::cout << "Loaded " << images.numImages << " images of size "
              << images.numRows << "x" << images.numCols << std::endl;
    std::cout << "Loaded " << labels.numLabels << " labels" << std::endl;

    // Display the first image
    int index = 0;
    std::cout << "Label: " << static_cast<int>(labels.getLabel(index)) << std::endl;
    images.displayImage(index);

    return 0;
}