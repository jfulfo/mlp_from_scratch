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
void softmax(float *x, float *result, int n);
float **matrix_softmax_activation(float **M, int n, int m);
float **matrix_activation(float **M, int m, int n, float (*activation)(float));
