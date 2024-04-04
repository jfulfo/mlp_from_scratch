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
void read_mnist(char *filename, float ***input_ptr, float ***target_ptr, int num_samples) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Could not open file %s\n", filename);
        return;
    }

    char line[2048]; // Assuming each line of the CSV does not exceed 2048 characters
    int sample_idx = 0;

    // Assuming the first line is not a header. If it is, uncomment the next line to skip it.
    // fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file) && sample_idx < num_samples) {
        char *token = strtok(line, ",");
        int target = atoi(token); // Convert first token to int for target
        
        // One-hot encode the target
        for (int i = 0; i < 10; i++) {
            (*target_ptr)[sample_idx][i] = (i == target) ? 1.0f : 0.0f;
        }

        // Process the rest of the tokens for the input
        for (int i = 0; i < 784; i++) {
            token = strtok(NULL, ",");
            if (token != NULL) {
                (*input_ptr)[sample_idx][i] = atoi(token) / 255.0f;
            }
        }
        sample_idx++;
    }

    fclose(file);
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
                printf("%f ", (*input_ptr)[i][j*28 + k]);
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
    read_mnist("mnist_train.csv", &input_ptr, &target_ptr, TRAINING_SAMPLES);

    int num_layers = 3;
    int num_neurons[] = {64, 32, 10};
    void (*activations[])(float*, float*, size_t) = {relu_vector, relu_vector, softmax};
    void (*activation_primes[])(float*, float*, size_t) = {relu_prime_vector, relu_prime_vector, softmax_prime};

    MLP *mlp = mlp_init(num_layers, num_neurons, activations, activation_primes, cross_entropy, softmax_ce_loss_prime, 0.005, 784);

    printf("Training...\n");
    int num_epochs = 10;
    train(mlp, input_ptr, target_ptr, num_epochs, TRAINING_SAMPLES, 32);
    print_mlp(mlp);

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
    read_mnist("mnist_test.csv", &input_ptr, &target_ptr, TEST_SAMPLES);
    validate(mlp, input_ptr, target_ptr, TEST_SAMPLES, 32);

    //printf("Final weights and biases:\n");
    //print_mlp(mlp);

    free(input_ptr);
    free(target_ptr);
    mlp_free(mlp);

    return 0;
}
