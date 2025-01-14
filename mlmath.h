#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm> // Add this include for std::max_element
#include <sstream>
#include <random> // Add this include at the top with other includes

namespace mlmath
{
    class Shape
    {
    public:
        unsigned int rows;
        unsigned int cols;
        Shape(unsigned int rows, unsigned int cols) : rows(rows), cols(cols) {}

        friend std::ostream &operator<<(std::ostream &os, const Shape &shape)
        {
            os << "(" << shape.rows << ", " << shape.cols << ")";
            return os;
        }

        // overload the == operator to compare two shapes
        bool operator==(const Shape &other) const
        {
            return rows == other.rows && cols == other.cols;
        }

        // overload the != operator to compare two shapes
        bool operator!=(const Shape &other) const
        {
            return !(*this == other);
        }
    };

    class Matrix
    {

    public:
        Shape shape;
        std::vector<std::vector<double>> data;

        Matrix(unsigned int rows, unsigned int cols) : shape(rows, cols)
        {
            data.resize(rows, std::vector<double>(cols, 0));
        }

        // Fix operator[] to return reference
        std::vector<double> &operator[](unsigned int i)
        {
            return data[i];
        }

        // Add const version of operator[]
        const std::vector<double> &operator[](unsigned int i) const
        {
            return data[i];
        }

        // define zeros functino that initalize all zeros matrix

        // Static factory methods
        static Matrix zeros(unsigned int rows, unsigned int cols)
        {
            return Matrix(rows, cols); // Matrix constructor already initializes with zeros
        }

        static Matrix ones(unsigned int rows, unsigned int cols)
        {
            Matrix result(rows, cols);
            for (unsigned int i = 0; i < rows; i++)
            {
                for (unsigned int j = 0; j < cols; j++)
                {
                    result.data[i][j] = 1.0;
                }
            }
            return result;
        }

        static Matrix random(unsigned int rows, unsigned int cols, double min_val, double max_val)
        {
            Matrix result(rows, cols);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dis(min_val, max_val);

            for (unsigned int i = 0; i < rows; i++)
            {
                for (unsigned int j = 0; j < cols; j++)
                {
                    result.data[i][j] = dis(gen);
                }
            }
            return result;
        }

        // overload the << operator to print the matrix
        friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix)
        {
            for (unsigned int i = 0; i < matrix.shape.rows; i++)
            {
                for (unsigned int j = 0; j < matrix.shape.cols; j++)
                {
                    os << matrix.data[i][j] << " ";
                }
                os << std::endl;
            }
            return os;
        }

        // add two matrices, should have same shape
        Matrix operator+(const Matrix &other) const
        {
            if (shape != other.shape)
            {
                std::stringstream ss;
                ss << "Matrix shapes do not match for addition: "
                   << shape << " and " << other.shape;
                throw std::invalid_argument(ss.str());
            }

            Matrix result(shape.rows, shape.cols);
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < shape.cols; j++)
                {
                    result[i][j] = data[i][j] + other.data[i][j];
                }
            }
            return result;
        }

        // multiply two matrices, should have compatible shapes
        Matrix operator*(const Matrix &other) const
        {
            if (shape.cols != other.shape.rows)
            {
                std::stringstream ss;
                ss << "Matrix shapes are not compatible for multiplication: "
                   << shape << " and " << other.shape;
                throw std::invalid_argument(ss.str());
            }

            Matrix result(shape.rows, other.shape.cols);
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < other.shape.cols; j++)
                {
                    for (unsigned int k = 0; k < shape.cols; k++)
                    {
                        result[i][j] += data[i][k] * other.data[k][j];
                    }
                }
            }
            return result;
        }

        // Matrix-Matrix element-wise multiplication
        Matrix elementWiseMultiply(const Matrix &other) const
        {
            if (shape != other.shape)
            {
                std::stringstream ss;
                ss << "Matrix shapes do not match for element-wise multiplication: "
                   << shape << " and " << other.shape;
                throw std::invalid_argument(ss.str());
            }

            Matrix result(shape.rows, shape.cols);
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < shape.cols; j++)
                {
                    result[i][j] = data[i][j] * other.data[i][j];
                }
            }
            return result;
        }

        // operator *=
        Matrix &operator*=(const Matrix &other)
        {
            *this = *this * other;
            return *this;
        }

        // matrix-scaler multiplication
        Matrix operator*(double scalar) const
        {
            Matrix result(shape.rows, shape.cols);
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < shape.cols; j++)
                {
                    result[i][j] = data[i][j] * scalar;
                }
            }
            return result;
        }

        // operator *=
        Matrix &operator*=(double scalar)
        {
            *this = *this * scalar;
            return *this;
        }

        // vector-matrix multiplication (dot product) should use dot function to be clear
        std::vector<double> dot(const std::vector<double> &vector) const
        {
            if (shape.cols != vector.size())
            {
                std::stringstream ss;
                ss << "Matrix and vector shapes are not compatible for dot product: "
                   << shape << " and (" << vector.size() << ", 1)";
                throw std::invalid_argument(ss.str());
            }

            std::vector<double> result(shape.rows, 0);
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < shape.cols; j++)
                {
                    result[i] += data[i][j] * vector[j];
                }
            }
            return result;
        }

        // operator^ element-wise power
        Matrix operator^(double scalar) const
        {
            Matrix result(shape.rows, shape.cols);
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < shape.cols; j++)
                {
                    result[i][j] = std::pow(data[i][j], scalar);
                }
            }
            return result;
        }

        // operator ^= element-wise power
        Matrix &operator^=(double scalar)
        {
            *this = *this ^ scalar;
            return *this;
        }

        // operator +=
        Matrix &operator+=(const Matrix &other)
        {
            *this = *this + other;
            return *this;
        }

        // add a scalar to the matrix
        Matrix operator+(double scalar) const
        {
            Matrix result(shape.rows, shape.cols);
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < shape.cols; j++)
                {
                    result[i][j] = data[i][j] + scalar;
                }
            }
            return result;
        }

        // operator +=
        Matrix &operator+=(double scalar)
        {
            *this = *this + scalar;
            return *this;
        }

        // operator- (reuse the muliply by scalar and add)
        Matrix operator-(const Matrix &other) const
        {
            return *this + (other * -1);
        }

        // operator -=
        Matrix &operator-=(const Matrix &other)
        {
            *this = *this - other;
            return *this;
        }

        // operator- (reuse the add scalar)
        Matrix operator-(double scalar) const
        {
            return *this + (-scalar);
        }

        // operator -=
        Matrix &operator-=(double scalar)
        {
            *this = *this - scalar;
            return *this;
        }

        // operator- (reuse the add scalar)
        Matrix operator-() const
        {
            return *this * -1;
        }

        // Add right-hand scalar division
        Matrix operator/(double scalar)
        {
            if (scalar == 0)
            {
                throw std::invalid_argument("Cannot divide by zero");
            }

            Matrix result(shape.rows, shape.cols);
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < shape.cols; j++)
                {
                    result[i][j] = data[i][j] / scalar;
                }
            }

            return result;
        }

        // operator /=
        Matrix &operator/=(double scalar)
        {
            *this = *this / scalar;
            return *this;
        }
        // transpose the matrix
        Matrix transpose() const
        {
            Matrix result(shape.cols, shape.rows);
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < shape.cols; j++)
                {
                    result.data[j][i] = data[i][j]; // Fixed indices were swapped
                }
            }
            return result;
        }

        // reshape the matrix
        Matrix reshape(unsigned int rows, unsigned int cols) const
        {
            if (rows * cols != shape.rows * shape.cols)
            {
                std::stringstream ss;
                ss << "Cannot reshape matrix of shape " << shape << " to shape (" << rows << ", " << cols << ")";
                throw std::invalid_argument(ss.str());
            }

            Matrix result(rows, cols);
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < shape.cols; j++)
                {
                    unsigned int index = i * shape.cols + j;
                    result.data[index / cols][index % cols] = data[i][j];
                }
            }
            return result;
        }

        double sum() const
        {
            double result = 0;
            for (unsigned int i = 0; i < shape.rows; i++)
            {
                for (unsigned int j = 0; j < shape.cols; j++)
                {
                    result += data[i][j];
                }
            }
            return result;
        }
    };

    // argmax and argmin of a vector
    double argmax(const std::vector<double> &vector)
    {
        if (vector.size() == 0)
        {
            throw std::invalid_argument("Cannot find argmax of an empty vector");
        }

        return std::distance(vector.begin(), std::max_element(vector.begin(), vector.end()));
    }

    double argmin(const std::vector<double> &vector)
    {
        if (vector.size() == 0)
        {
            throw std::invalid_argument("Cannot find argmin of an empty vector");
        }

        return std::distance(vector.begin(), std::min_element(vector.begin(), vector.end()));
    }

    // argmax and argmin for matrix
    double argmax(const Matrix &matrix)
    {
        if (matrix.shape.rows == 0 || matrix.shape.cols == 0)
        {
            throw std::invalid_argument("Cannot find argmax of an empty matrix");
        }

        double max_value = matrix.data[0][0];
        unsigned int max_index = 0;
        for (unsigned int i = 0; i < matrix.shape.rows; i++)
        {
            for (unsigned int j = 0; j < matrix.shape.cols; j++)
            {
                if (matrix.data[i][j] > max_value)
                {
                    max_value = matrix.data[i][j];
                    max_index = i * matrix.shape.cols + j;
                }
            }
        }
        return max_index;
    }

    double argmin(const Matrix &matrix)
    {
        if (matrix.shape.rows == 0 || matrix.shape.cols == 0)
        {
            throw std::invalid_argument("Cannot find argmin of an empty matrix");
        }

        double min_value = matrix.data[0][0];
        unsigned int min_index = 0;
        for (unsigned int i = 0; i < matrix.shape.rows; i++)
        {
            for (unsigned int j = 0; j < matrix.shape.cols; j++)
            {
                if (matrix.data[i][j] < min_value)
                {
                    min_value = matrix.data[i][j];
                    min_index = i * matrix.shape.cols + j;
                }
            }
        }
        return min_index;
    }

    // relu and relu derivative of scalar, vector and matrix
    double relu(double x)
    {
        return std::max(0.0, x);
    }

    double relu_derivative(double x)
    {
        return x > 0 ? 1 : 0;
    }

    std::vector<double> relu(const std::vector<double> &vector)
    {
        std::vector<double> result(vector.size());
        for (unsigned int i = 0; i < vector.size(); i++)
        {
            result[i] = relu(vector[i]);
        }
        return result;
    }

    std::vector<double> relu_derivative(const std::vector<double> &vector)
    {
        std::vector<double> result(vector.size());
        for (unsigned int i = 0; i < vector.size(); i++)
        {
            result[i] = relu_derivative(vector[i]);
        }
        return result;
    }

    Matrix relu(const Matrix &matrix)
    {
        Matrix result(matrix.shape.rows, matrix.shape.cols);
        for (unsigned int i = 0; i < matrix.shape.rows; i++)
        {
            for (unsigned int j = 0; j < matrix.shape.cols; j++)
            {
                result[i][j] = relu(matrix[i][j]);
            }
        }
        return result;
    }

    Matrix relu_derivative(const Matrix &matrix)
    {
        Matrix result(matrix.shape.rows, matrix.shape.cols);
        for (unsigned int i = 0; i < matrix.shape.rows; i++)
        {
            for (unsigned int j = 0; j < matrix.shape.cols; j++)
            {
                result[i][j] = relu_derivative(matrix[i][j]);
            }
        }
        return result;
    }

}