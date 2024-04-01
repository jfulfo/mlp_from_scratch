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


MLP *mlp_init(int num_layers, int *num_neurons, float (*activation[])(float), float (*activation_prime[])(float), 
            float (*loss)(float, float), float learning_rate, int input_size, int output_size) {
    srand(time(NULL));
    MLP *mlp = malloc(sizeof(MLP));
    mlp->layers = malloc(num_layers * sizeof(Layer));   
    for (int i = 0; i < num_layers; i++) {
        Layer *layer = malloc(sizeof(Layer));
        layer->num_neurons = num_neurons[i];
        layer->prev_num_neurons = i == 0 ? input_size : num_neurons[i - 1];
        layer->weights = malloc(layer->num_neurons * sizeof(float *));
        layer->biases = malloc(layer->num_neurons * sizeof(float));
        layer->activation = activation[i];
        layer->activation_prime = activation_prime[i];
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
    mlp->learning_rate = learning_rate;
    mlp->input_size = input_size;
    mlp->output_size = output_size;
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

float **batch_output(MLP *mlp, float **input, int batch_size) {
    return batch_forward(mlp, input, batch_size)[mlp->num_layers];
}

float ***batch_forward(MLP *mlp, float **inputs, int batch_size) {
    float ***activations = malloc((mlp->num_layers + 2) * sizeof(float **));
    for (int i = 0; i <= mlp->num_layers; i++) {
        activations[i] = allocate_matrix((i == 0) ? batch_size : mlp->layers[i-1]->num_neurons, (i == 0) ? mlp->input_size : mlp->layers[i-1]->num_neurons);
    }
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < mlp->input_size; j++) {
            activations[0][i][j] = inputs[i][j];
        }
    }
    int activation_size = mlp->input_size;
    for (int i = 0; i < mlp->num_layers; i++) {
        Layer *layer = mlp->layers[i];
        float **output = gemm_add(activations[i], layer->weights, batch_size, activation_size, layer->num_neurons, layer->biases); 
        if (layer->activation == softmax) {
            activations[i + 1] = matrix_softmax_activation(output, batch_size, layer->num_neurons);
        }
        else {
            activations[i + 1] = matrix_activation(output, batch_size, layer->num_neurons, mlp->layers[i]->activation);
        }
        activation_size = layer->num_neurons;
    }
    // activate output layer
    for (int i = 0; i < batch_size; i++) {
        softmax(activations[mlp->num_layers][i], activations[mlp->num_layers][i], mlp->output_size);
    }
    for (int i = 0; i <= mlp->num_layers; i++) {
        print_matrix(activations[i], batch_size, (i == 0) ? mlp->input_size : mlp->layers[i-1]->num_neurons);
    }
    return activations;
}

void calculate_gradient_and_update(MLP *mlp, float **deltas, float **prev_activations, int batch_size, int layer_idx) {
    int num_neurons = mlp->layers[layer_idx]->num_neurons;
    int prev_num_neurons = layer_idx == 0 ? mlp->input_size : mlp->layers[layer_idx - 1]->num_neurons;

    float **prev_activations_T = transpose(prev_activations, batch_size, prev_num_neurons);
    float **grad_weights = gemm(prev_activations_T, deltas, num_neurons, prev_num_neurons, batch_size);

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
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < num_neurons; j++) {
                    new_deltas[i][j] = mlp->loss(activations[layer_idx][i][j], targets[i][j]) * mlp->layers[layer_idx]->activation_prime(activations[layer_idx][i][j]);
                }
            }
        } else {
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < num_neurons; j++) {
                    float sum = 0;
                    for (int k = 0; k < mlp->layers[layer_idx + 1]->num_neurons; k++) {
                        sum += deltas[i][k] * mlp->layers[layer_idx + 1]->weights[k][j];
                    }
                    new_deltas[i][j] = sum * mlp->layers[layer_idx]->activation_prime(activations[layer_idx][i][j]);
                }
            }
        }

        if (deltas != NULL) free_matrix(deltas, batch_size);
        deltas = new_deltas;

        float **prev_activations = layer_idx == 0 ? inputs : activations[layer_idx - 1];
        calculate_gradient_and_update(mlp, deltas, prev_activations, batch_size, layer_idx);
    }

    // free activations
    for (int i = 0; i <= mlp->num_layers; i++) {
        free_matrix(activations[i], batch_size);
    }
    free(activations);
    free_matrix(deltas, batch_size);
}

void train(MLP *mlp, float **inputs, float **targets, int num_epochs, int num_samples, int batch_size) {
    float loss;
    int num_batches = num_samples / batch_size;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("Epoch %d\n", epoch+1);
        float **batch_inputs = malloc(batch_size * sizeof(float *));
        float **batch_targets = malloc(batch_size * sizeof(float *));
        for (int i = 0; i < batch_size; i++) {
            batch_inputs[i] = inputs[i];
            batch_targets[i] = targets[i];
        }
        batch_backward(mlp, batch_inputs, batch_targets, batch_size);
        free(batch_inputs);
        free(batch_targets);
        loss = validate(mlp, inputs, targets, num_samples);
    }
}

float validate(MLP *mlp, float **inputs, float **targets, int num_samples) {
    float **outputs = batch_output(mlp, inputs, num_samples);
    float loss = 0;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < mlp->output_size; j++) {
            loss += mlp->loss(outputs[i][j], targets[i][j]);
        }
    }
    printf("Loss: %f\n", loss / (float)num_samples);
    printf("Accuracy: %f\n", accuracy(mlp, inputs, targets, num_samples));
    /*
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < mlp->output_size; j++) {
            printf("Target: %f, Output: %f\n", targets[i][j], outputs[i][j]);
        }
    }
    */
    free_matrix(outputs, num_samples);
    return loss;
}

float accuracy(MLP *mlp, float **inputs, float **targets, int num_samples) {
    float **outputs = batch_output(mlp, inputs, num_samples);
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        int max_idx = 0;
        for (int j = 1; j < mlp->output_size; j++) {
            if (outputs[i][j] > outputs[i][max_idx]) {
                max_idx = j;
            }
        }
        if (targets[i][max_idx] == 1) {
            correct++;
        }
    }
    free_matrix(outputs, num_samples);
    return (float)correct / (float)num_samples;
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
