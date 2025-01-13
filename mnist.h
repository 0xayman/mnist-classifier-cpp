#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace mnist
{
    class MNISTReader
    {
    protected:
        // Read 4 bytes from the file and convert them to an integer
        static int readInt(std::ifstream &file)
        {
            unsigned char bytes[4];
            file.read(reinterpret_cast<char *>(bytes), 4);
            return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
        }
    };

    class MNISTImages : public MNISTReader
    {
    public:
        int numImages;
        int numRows;
        int numCols;

        std::vector<std::vector<unsigned char>> images;

        MNISTImages(const std::string &filename)
        {
            loadImages(filename);
        }

        void loadImages(const std::string &filename)
        {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Cannot open file `" + filename + "`");
            }

            int magicNumber = readInt(file); // read the first 4 bytes (magicNumber) from the file
            // the magic number should be hex 0x00000803 which is 2051 in decimal
            if (magicNumber != 0x00000803)
            {
                throw std::runtime_error("Invalid MNIST image file!");
            }

            numImages = readInt(file); // read the second 4 bytes (numImages) from the file
            numRows = readInt(file);   // read the third 4 bytes (numRows) from the file
            numCols = readInt(file);   // read the fourth 4 bytes (numCols) from the file

            // Read the image data
            int imageSize = numRows * numCols;
            images.resize(numImages, std::vector<unsigned char>(imageSize));
            for (int i = 0; i < numImages; i++)
            {
                file.read(reinterpret_cast<char *>(images[i].data()), imageSize);
            }
        }

        void displayImage(int index) const
        {
            if (index < 0 || index >= numImages)
            {
                throw std::out_of_range("Invalid image index");
            }

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    std::cout << (images[index][i * numCols + j] > 128 ? "#" : ".");
                }
                std::cout << std::endl;
            }
        }
    };

    class MNISTLabels : public MNISTReader
    {
    public:
        int numLabels;
        std::vector<unsigned char> labels;

        MNISTLabels(const std::string &filename)
        {
            loadLabels(filename);
        }

        void loadLabels(const std::string &filename)
        {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open())
            {
                throw std::runtime_error("Cannot open file `" + filename + "`");
            }

            int magicNumber = readInt(file); // read the first 4 bytes (magicNumber) from the file
            // the magic number should be hex 0x00000801 which is 2049 in decimal
            if (magicNumber != 0x00000801)
            {
                throw std::runtime_error("Invalid MNIST label file!");
            }

            numLabels = readInt(file); // read the second 4 bytes (numLabels) from the file
            labels.resize(numLabels);
            file.read(reinterpret_cast<char *>(labels.data()), numLabels);
        }

        unsigned char getLabel(int index) const
        {
            if (index < 0 || index >= numLabels)
            {
                throw std::out_of_range("Invalid label index");
            }
            return labels[index];
        }
    };

}