#include<stdlib.h>
#include<stdio.h>
#include"gemm.h"
/*
    Computes A @ B
    Parameters:
    A: m x k matrix
    B: k x n matrix
    m: number of rows of A
    n: number of columns of B
    k: number of columns of A and rows of B
*/
float **gemm(float **A, float **B, int m, int n, int k) {
    float **C = allocate_matrix(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int l = 0; l < k; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
    return C;
}

// computes C = AB + x
float **gemm_add(float **A, float **B, int m, int n, int k, float *x) {
    float **C = gemm(A, B, m, n, k);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] += x[j];
        }
    }
    return C;
}

float **transpose(float **M, int m, int n) {
    float **T = allocate_matrix(n, m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            T[j][i] = M[i][j];
        }
    }
    return T;
}

float **allocate_matrix(int m, int n) {
    float **M = (float **)malloc(m * sizeof(float *));
    for (int i = 0; i < m; i++) {
        M[i] = (float *)malloc(n * sizeof(float));
    }
    return M;
}

void free_matrix(float **M, int m) {
    for (int i = 0; i < m; i++) {
        free(M[i]);
    }
    free(M);
}

void print_matrix(float **M, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", M[i][j]);
        }
        printf("\n");
    }
}

// take the max value of the vector and set it to 1, all others to 0
void one_hot_vector(float *v, int n) {
    float max_val = v[0];
    int max_index = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > max_val) {
            max_val = v[i];
            max_index = i;
        }
    }
    for (int i = 0; i < n; i++) {
        v[i] = i == max_index ? 1 : 0;
    }
}