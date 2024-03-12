#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class Matrix {
public:
    Matrix(int rows, int cols);

    int getElement(int row, int col) const;
    void setElement(int row, int col, int value);

    bool isInvertible() const;

private:
    std::vector<std::vector<int>> data;

    int determinant(const std::vector<std::vector<int>>& mat) const;
};

#endif // MATRIX_H
