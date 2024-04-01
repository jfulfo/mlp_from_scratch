#include"loss.h"

float mse(float y, float y_hat) {
    return 0.5 * (y - y_hat) * (y - y_hat);
}

float cross_entropy(float y, float y_hat) {
    if (y_hat == 0.0) y_hat += 1e-15;
    if (y_hat == 1.0) y_hat -= 1e-15;

    float loss = -1.0 * (y * log(y_hat) + (1.0 - y) * log(1.0 - y_hat));
    return loss;
}