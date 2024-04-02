/*
Multi-layer perceptron
from scratch, in C
~ Jamie
*/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"mlp.h"
#include"activation.h"
#include"loss.h"
#include"gemm.h"


MLP *mlp_init(int num_layers, int *num_neurons, void (*activations[])(float*, float*, size_t), void (*activations_prime[])(float*, float*, size_t),
            float (*loss)(float, float), void (*loss_prime)(float**, float**, float**, int, int), float learning_rate, int input_size) {
    srand(time(NULL));
    MLP *mlp = malloc(sizeof(MLP));
    mlp->layers = malloc(num_layers * sizeof(Layer));   
    for (int i = 0; i < num_layers; i++) {
        Layer *layer = malloc(sizeof(Layer));
        layer->num_neurons = num_neurons[i];
        layer->prev_num_neurons = i == 0 ? input_size : num_neurons[i - 1];
        layer->weights = malloc(layer->num_neurons * sizeof(float *));
        layer->biases = malloc(layer->num_neurons * sizeof(float));
        layer->activation = activations[i];
        layer->activation_prime = activations_prime[i];
        for (int j = 0; j < num_neurons[i]; j++) {
            layer->weights[j] = malloc(layer->prev_num_neurons * sizeof(float));
            for (int k = 0; k < layer->prev_num_neurons; k++) {
                layer->weights[j][k] = (float)rand() / (float)RAND_MAX;
            }
            layer->biases[j] = (float)rand() / (float)RAND_MAX;
        }
        mlp->layers[i] = layer;
    }
    mlp->num_layers = num_layers;
    mlp->loss = loss;
    mlp->loss_prime = loss_prime;
    mlp->learning_rate = learning_rate;
    mlp->input_size = input_size;
    return mlp;
}

void mlp_free(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        Layer *layer = mlp->layers[i];
        free_matrix(layer->weights, layer->num_neurons);
        free(layer->biases);
        free(layer);
    }
    free(mlp->layers);
    free(mlp);
}

float ***batch_forward(MLP *mlp, float **inputs, int batch_size) {
    float ***activations = malloc((mlp->num_layers + 1) * sizeof(float **));
    activations[0] = allocate_matrix(batch_size, mlp->input_size);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < mlp->input_size; j++) {
            activations[0][i][j] = inputs[i][j];
        }
    }
    for (int i = 1; i <= mlp->num_layers; i++) {
        activations[i] = allocate_matrix(batch_size, mlp->layers[i-1]->num_neurons);
    }

    for (int i = 0; i < mlp->num_layers; i++) {
        Layer *layer = mlp->layers[i];
        float **output = gemm_add(activations[i], layer->weights, batch_size, layer->prev_num_neurons, layer->num_neurons, layer->biases); 
        activations[i + 1] = matrix_activation(output, batch_size, layer->num_neurons, mlp->layers[i]->activation);
    }
    return activations;
}

void calculate_gradient_and_update(MLP *mlp, float **deltas, float **prev_activations, int batch_size, int layer_idx) {
    int num_neurons = mlp->layers[layer_idx]->num_neurons;
    int prev_num_neurons = mlp->layers[layer_idx]->prev_num_neurons;

    float **prev_activations_T = transpose(prev_activations, batch_size, prev_num_neurons);
    float **grad_weights = gemm(prev_activations_T, deltas, num_neurons, prev_num_neurons, batch_size);
    free_matrix(prev_activations_T, prev_num_neurons);

    float *grad_biases = malloc(num_neurons * sizeof(float));
    for (int i = 0; i < num_neurons; i++) {
        grad_biases[i] = 0;
        for (int j = 0; j < batch_size; j++) {
            grad_biases[i] += deltas[j][i];
        }
    }

    for (int i = 0; i < num_neurons; i++) {
        for (int j = 0; j < prev_num_neurons; j++) {
            mlp->layers[layer_idx]->weights[i][j] -= mlp->learning_rate * grad_weights[i][j] / batch_size;
        }
        mlp->layers[layer_idx]->biases[i] -= mlp->learning_rate * grad_biases[i] / batch_size;
    }

    free_matrix(grad_weights, num_neurons);
    free(grad_biases);
}

void batch_backward(MLP *mlp, float **inputs, float **targets, int batch_size) {
    float ***activations = batch_forward(mlp, inputs, batch_size);
    float **deltas = NULL;
    for (int layer_idx = mlp->num_layers - 1; layer_idx >= 0; layer_idx--) {
        Layer *layer = mlp->layers[layer_idx];
        int num_neurons = layer->num_neurons;

        float **new_deltas = allocate_matrix(batch_size, num_neurons);
        if (layer_idx == mlp->num_layers - 1) {
            mlp->loss_prime(activations[mlp->num_layers], targets, new_deltas, batch_size, num_neurons);
        } else {
            float **weights_T = transpose(layer->weights, num_neurons, layer->prev_num_neurons);
            float **propagated_deltas = gemm(new_deltas, weights_T, batch_size, layer->prev_num_neurons, num_neurons);
            new_deltas = matrix_activation(propagated_deltas, batch_size, num_neurons, layer->activation_prime);
            free_matrix(weights_T, num_neurons);

        }
        if (deltas != NULL) free_matrix(deltas, batch_size);
        deltas = new_deltas;

        float **prev_activations = layer_idx == 0 ? inputs : activations[layer_idx - 1];
        calculate_gradient_and_update(mlp, deltas, prev_activations, batch_size, layer_idx);
    }

    for (int i = 0; i <= mlp->num_layers; i++) {
        free_matrix(activations[i], batch_size);
    }
    free(activations);
    free_matrix(deltas, batch_size);
}

void train(MLP *mlp, float **inputs, float **targets, int num_epochs, int num_samples, int batch_size) {
    int num_batches = num_samples / batch_size;
    int output_size = mlp->layers[mlp->num_layers-1]->num_neurons;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("\nEpoch %d\n", epoch+1);
        for (int i = 0; i < num_batches; i++) {
            print_progress(i, num_batches);
            float **batch_inputs = malloc(batch_size * sizeof(float *));
            float **batch_targets = malloc(batch_size * sizeof(float *));
            for (int j = 0; j < batch_size; j++) {
                batch_inputs[j] = malloc(mlp->input_size * sizeof(float));
                batch_targets[j] = malloc(output_size * sizeof(float));
                for (int k = 0; k < mlp->input_size; k++) {
                    batch_inputs[j][k] = inputs[i * batch_size + j][k];
                }
                for (int k = 0; k < output_size; k++) {
                    batch_targets[j][k] = targets[i * batch_size + j][k];
                }
            }
            batch_backward(mlp, batch_inputs, batch_targets, batch_size); 
            mlp->learning_rate *= 0.99;
            free_matrix(batch_inputs, batch_size);
            free_matrix(batch_targets, batch_size);
        }
        print_progress(num_batches, num_batches);
        printf("\n");
        validate(mlp, inputs, targets, num_samples, batch_size);
    }
}

void validate(MLP *mlp, float **inputs, float **targets, int num_samples, int batch_size) {
    float loss = 0;
    int correct = 0;
    int num_batches = num_samples / batch_size;
    int output_size = mlp->layers[mlp->num_layers-1]->num_neurons;
    for (int i = 0; i < num_batches; i++) {
        float **batch_inputs = malloc(batch_size * sizeof(float *));
        float **batch_targets = malloc(batch_size * sizeof(float *));
        for (int j = 0; j < batch_size; j++) {
            batch_inputs[j] = malloc(mlp->input_size * sizeof(float));
            batch_targets[j] = malloc(output_size * sizeof(float));
            for (int k = 0; k < mlp->input_size; k++) {
                batch_inputs[j][k] = inputs[i * batch_size + j][k];
            }
            for (int k = 0; k < output_size; k++) {
                batch_targets[j][k] = targets[i * batch_size + j][k];
            }
        }

        float ***activations = batch_forward(mlp, batch_inputs, batch_size);
        float **outputs = activations[mlp->num_layers];
        for (int j = 0; j < batch_size; j++) {
            for (int k = 0; k < output_size; k++) {
                loss += mlp->loss(outputs[j][k], batch_targets[j][k]);
            }
        }
        for (int j = 0; j < batch_size; j++) {
            float max_val = outputs[j][0];
            int max_idx = 0;
            for (int k = 1; k < output_size; k++) {
                if (outputs[j][k] > max_val) {
                    max_val = outputs[j][k];
                    max_idx = k;
                }
            }
            for (int k = 0; k < output_size; k++) {
                if (k == max_idx && batch_targets[j][k] == 1.0) {
                    correct++;
                    break;
                }
            }
        }
        
        free_matrix(batch_inputs, batch_size);
        free_matrix(batch_targets, batch_size);
        for (int j = 0; j <= mlp->num_layers; j++) {
            free_matrix(activations[j], batch_size);
        }
        free(activations);
    }

    printf("Loss: %f\n", loss / (float)num_samples);
    printf("Accuracy: %f\n", (float)correct / (float)num_samples);
}

void print_progress(int current_step, int total_steps) {
    int bar_width = 100;
    float progress = (float)current_step / total_steps;
    int pos = bar_width * progress;

    printf("\rEpoch progress: [");
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) printf("#");
        else printf(" ");
    }
    printf("] %d%%", (int)(progress * 100));
    fflush(stdout);
}

void print_mlp(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        printf("Layer %d:\n", i);
        for (int j = 0; j < mlp->layers[i]->num_neurons; j++) {
            printf("\tNeuron %d:\n", j);
            for (int k = 0; k < mlp->layers[i]->prev_num_neurons; k++) {
                printf("\t\tWeight %d: %f\n", k, mlp->layers[i]->weights[j][k]);
            }
            printf("\t\tBias: %f\n", mlp->layers[i]->biases[j]);
        }
    }
}
