#include<math.h>

void softmax_ce_loss_prime(float **outputs, float **targets, float **deltas, int batch_size, int num_neurons);
float mse(float y, float y_hat);
float cross_entropy(float y, float y_hat);
