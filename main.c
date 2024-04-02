/*
    Demo using MLP to solve XOR
*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"mlp.h"
#include"activation.h"
#include"loss.h"
#include"gemm.h"

#define TRAINING_SAMPLES 60000
#define TEST_SAMPLES 10000

// for first number is the target, the rest are the input
// params: filename, pointer to matrix of input, pointer to matrix of target, number of samples
void read_mnist(char *filename, float ***input_ptr, float ***target_ptr) {
    FILE *file = fopen(filename, "r");
    char *line = NULL;
    size_t len = 0;
    int sample_idx = 0;
    while (getline(&line, &len, file) != -1) {
        int target = atoi(&line[0]);
        for (int i = 0; i < 10; i++) {
            (*target_ptr)[sample_idx][i] = i == target ? 1.0 : 0.0;
        }
        for (int i = 2; i < 785*2; i += 2) {
            (*input_ptr)[sample_idx][(i - 2) / 2] = atof(&line[i]) / 255.0;
        }
        sample_idx++;
    }
}

void view_mnist(float ***input_ptr, float ***target_ptr, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        printf("Target: ");
        for (int j = 0; j < 10; j++) {
            printf("%.0f ", (*target_ptr)[i][j]);
        }
        printf("\n");
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                printf("%.0f ", (*input_ptr)[i][j * 28 + k] * 255);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main() {

    float **input_ptr = malloc(TRAINING_SAMPLES * sizeof(float *));
    for (int i = 0; i < TRAINING_SAMPLES; i++) {
        input_ptr[i] = malloc(784 * sizeof(float));
    }
    float **target_ptr = malloc(TRAINING_SAMPLES * sizeof(float *));
    for (int i = 0; i < TRAINING_SAMPLES; i++) {
        target_ptr[i] = malloc(10 * sizeof(float));
    }
    read_mnist("mnist_train.csv", &input_ptr, &target_ptr);

    int num_layers = 3;
    int num_neurons[] = {128, 64, 10};
    void (*activations[])(float*, float*, size_t) = {relu_vector, relu_vector, softmax};
    void (*activation_primes[])(float*, float*, size_t) = {relu_prime_vector, relu_prime_vector, softmax_prime};

    MLP *mlp = mlp_init(num_layers, num_neurons, activations, activation_primes, cross_entropy, softmax_ce_loss_prime, 0.01, 784);

    printf("Training...\n");
    int num_epochs = 10;
    train(mlp, input_ptr, target_ptr, num_epochs, TRAINING_SAMPLES, 32);

    free_matrix(input_ptr, TRAINING_SAMPLES);
    free_matrix(target_ptr, TRAINING_SAMPLES);
    input_ptr = malloc(TEST_SAMPLES * sizeof(float *));
    for (int i = 0; i < TEST_SAMPLES; i++) {
        input_ptr[i] = malloc(784 * sizeof(float));
    }
    target_ptr = malloc(TEST_SAMPLES * sizeof(float *));
    for (int i = 0; i < TEST_SAMPLES; i++) {
        target_ptr[i] = malloc(10 * sizeof(float));
    }
    read_mnist("mnist_test.csv", &input_ptr, &target_ptr);
    validate(mlp, input_ptr, target_ptr, TEST_SAMPLES, 32);

    //printf("Final weights and biases:\n");
    //print_mlp(mlp);

    free(input_ptr);
    free(target_ptr);
    mlp_free(mlp);

    return 0;
}
