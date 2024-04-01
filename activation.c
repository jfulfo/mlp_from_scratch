#include<math.h>
#include<stdlib.h>
#include"activation.h"

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float sigmoid_prime(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
} 

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_prime(float x) {
    return x > 0 ? 1 : 0;
}

float leaky_relu(float x) {
    return x > 0 ? x : 0.01 * x;
}

float leaky_relu_prime(float x) {
    return x > 0 ? 1 : 0.01;
}

float tanh_float(float x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

float tanh_prime(float x) {
    return 1 - tanh(x) * tanh(x);
}

void softmax(float *x, float *result, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        result[i] = exp(x[i]);
        sum += result[i];    
    }
    for (int i = 0; i < n; i++) {
        result[i] /= sum;
    }
}

float **matrix_softmax_activation(float **M, int n, int m) {
    for (int i = 0; i < n; i++) {
        softmax(M[i], M[i], m);
    }
    return M;
}

float **matrix_activation(float **M, int n, int m, float (*activation)(float)) {
    if (activation == softmax) {
        return matrix_softmax_activation(M, n, m);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            M[i][j] = activation(M[i][j]);
        }
    }
    return M;
}