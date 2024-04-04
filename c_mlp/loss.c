#include"loss.h"

void softmax_ce_loss_prime(float **outputs, float **targets, float **deltas, int batch_size, int num_neurons) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_neurons; j++) {
            deltas[i][j] = outputs[i][j] - targets[i][j];
        }
    }
}

float mse(float y, float y_hat) {
    return 0.5 * (y - y_hat) * (y - y_hat);
}

float cross_entropy(float y, float y_hat) {
    if (y_hat == 0) {
        y_hat = 0.00001;
    } else if (y_hat == 1) {
        y_hat = 0.99999;
    }

    float loss = -1.0 * (y * log(y_hat) + (1.0 - y) * log(1.0 - y_hat));
    return loss;
}
