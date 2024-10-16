#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std;


void matrixMultiplyJIK(int** A, int** B, int** C, int n) {
#pragma omp parallel for schedule(guided, 4)
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrixMultiplyKIJ(int** A, int** B, int** C, int n) {
#pragma omp parallel for schedule(guided, 4)
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void filler(int** matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = rand()%100;
        }
    }
}


void printMatrix(int** matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    setlocale(LC_ALL, "RU");
    omp_set_num_threads(2);
    int n = 1024;  
    int** A = new int* [n];
    int** B = new int* [n];
    int** C = new int* [n];
    int** C2 = new int* [n];
    for (int i = 0; i < n; ++i) {
        A[i] = new int[n];
        B[i] = new int[n];
        C[i] = new int[n];
        C2[i] = new int[n];
    }

    filler(A, n);
    filler(B, n);


    auto startJIK = chrono::high_resolution_clock::now();
    matrixMultiplyJIK(A, B, C, n);
    auto endJIK = chrono::high_resolution_clock::now();
    chrono::duration<double> timeJIK = endJIK - startJIK;
    cout << "Время работы с циклом j-i-k: " << timeJIK.count() << " секунд" << endl;

    auto startKIJ = chrono::high_resolution_clock::now();
    matrixMultiplyKIJ(A, B, C2, n);
    auto endKIJ = chrono::high_resolution_clock::now();
    chrono::duration<double> timeKIJ = endKIJ - startKIJ;
    cout << "Время работы с циклом k-i-j: " << timeKIJ.count() << " секунд" << endl;

    for (int i = 0; i < n; ++i) {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
        delete[] C2[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C2;

    return 0;
}