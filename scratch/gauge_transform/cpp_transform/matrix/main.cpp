#include <iostream>
#include "Matrix.h"

int main() {
    // Create a matrix with 3 rows and 3 columns
    Matrix myMatrix(3, 3);

    // Set values in the matrix
    myMatrix.setElement(0, 0, 1);
    myMatrix.setElement(0, 1, 2);
    myMatrix.setElement(0, 2, 3);
    myMatrix.setElement(1, 0, 4);
    myMatrix.setElement(1, 1, 5);
    myMatrix.setElement(1, 2, 6);
    myMatrix.setElement(2, 0, 7);
    myMatrix.setElement(2, 1, 8);
    myMatrix.setElement(2, 2, 9);

    // Display the matrix
    std::cout << "Matrix:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << myMatrix.getElement(i, j) << ' ';
        }
        std::cout << std::endl;
    }

    // Check if the matrix is invertible
    if (myMatrix.isInvertible()) {
        std::cout << "The matrix is invertible." << std::endl;
    } else {
        std::cout << "The matrix is not invertible." << std::endl;
    }

    return 0;
}
