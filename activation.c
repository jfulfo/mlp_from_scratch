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

void relu_vector(float *input, float *output, size_t len) {
    for (int i = 0; i < len; i++) {
        output[i] = relu(input[i]);
    }
}

void relu_prime_vector(float *input, float *output, size_t len) {
    for (int i = 0; i < len; i++) {
        output[i] = relu_prime(input[i]);
    }
}

void softmax(float *input, float *output, size_t len) {
    float max = input[0];
    float sum = 0.0;

    for (int i = 1; i < len; ++i) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    for (int i = 0; i < len; ++i) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }
    for (int i = 0; i < len; ++i) {
        output[i] /= sum;
    }
}

void softmax_prime(float *input, float *output, size_t len) {
    for (int i = 0; i < len; i++) {
        output[i] = input[i] * (1 - input[i]);
    }
}


float **matrix_activation(float **M, int n, int m, void (*activation)(float*, float*, size_t)) {
    for (int i = 0; i < n; i++) {
        activation(M[i], M[i], m); 
    }
    return M;
}