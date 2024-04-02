// this is just for linking the activation functions to the neural network
#include<math.h>

float sigmoid(float x);
float sigmoid_prime(float x);
float relu(float x);
float relu_prime(float x);
float leaky_relu(float x);
float leaky_relu_prime(float x);
float tanh_float(float x);
float tanh_prime(float x);
void relu_vector(float *input, float *output, size_t len);
void relu_prime_vector(float *input, float *output, size_t len);
void softmax(float *input, float *output, size_t len);
void softmax_prime(float *input, float *output, size_t len);
float **matrix_softmax_activation(float **M, int n, int m);
float **matrix_activation(float **M, int m, int n, void (*activation)(float*, float*, size_t));