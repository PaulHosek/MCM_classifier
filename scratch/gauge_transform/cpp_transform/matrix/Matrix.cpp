#include "Matrix.h"

Matrix::Matrix(int rows, int cols) : data(rows, std::vector<int>(cols, 0)) {}

int Matrix::getElement(int row, int col) const {
    return data[row][col];
}

void Matrix::setElement(int row, int col, int value) {
    data[row][col] = value;
}

bool Matrix::isInvertible() const {
    return determinant(data) != 0;
}

int Matrix::determinant(const std::vector<std::vector<int>>& mat) const {
    int size = mat.size();

    // Base case: 2x2 matrix
    if (size == 2) {
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    }

    int det = 0;
    for (int i = 0; i < size; ++i) {
        // Calculate the cofactor by excluding current row and column
        std::vector<std::vector<int>> submatrix(size - 1, std::vector<int>(size - 1, 0));

        for (int row = 1; row < size; ++row) {
            for (int col = 0, k = 0; col < size; ++col) {
                if (col != i) {
                    submatrix[row - 1][k++] = mat[row][col];
                }
            }
        }

        // Recursive call to calculate the determinant
        int sign = (i % 2 == 0) ? 1 : -1;
        det += sign * mat[0][i] * determinant(submatrix);
    }

    return det;
}
