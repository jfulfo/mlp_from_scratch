
typedef struct {
    float **weights;
    float *biases;
    int num_neurons;
    int prev_num_neurons;
    float (*activation)(float);
    float (*activation_prime)(float);
} Layer;

typedef struct {
    Layer **layers;
    int num_layers; // only hidden layers
    float (*loss)(float, float);
    float learning_rate;
    int input_size;
    int output_size;
} MLP;

Layer *layer_init(int num_neurons, int prev_num_neurons);
void layer_free(Layer *layer);
MLP *mlp_init(int num_layers, int *num_neurons, float (*activation[])(float), float (*activation_prime[])(float), 
            float (*loss)(float, float), float learning_rate, int input_size, int output_size);
void mlp_free(MLP *mlp);

float **batch_output(MLP *mlp, float **input, int batch_size);
float ***batch_forward(MLP *mlp, float **inputs, int batch_size);
void calculate_gradient_and_update(MLP *mlp, float **deltas, float **prev_activations, int batch_size, int layer_idx);
void batch_backward(MLP *mlp, float **inputs, float **targets, int batch_size);

void train(MLP *mlp, float **inputs, float **targets, int num_epochs, int num_samples, int batch_size);
float validate(MLP *mlp, float **inputs, float **targets, int num_samples);
float accuracy(MLP *mlp, float **inputs, float **targets, int num_samples);
void print_mlp(MLP *mlp);
