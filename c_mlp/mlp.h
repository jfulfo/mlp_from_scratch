
typedef struct {
    float **weights;
    float *biases;
    int num_neurons;
    int prev_num_neurons;
    void (*activation)(float*, float*, size_t);
    void (*activation_prime)(float*, float*, size_t);
} Layer;

typedef struct {
    Layer **layers;
    int num_layers; // only hidden layers and the output layer
    float (*loss)(float, float);
    void (*loss_prime)(float**, float**, float**, int, int);
    float learning_rate;
    int input_size;
} MLP;

Layer *layer_init(int num_neurons, int prev_num_neurons);
void layer_free(Layer *layer);
MLP *mlp_init(int num_layers, int *num_neurons, void (*activations[])(float*, float*, size_t), void (*activations_prime[])(float*, float*, size_t),
            float (*loss)(float, float), void (*loss_prime)(float**, float**, float**, int, int), float learning_rate, int input_size);
void mlp_free(MLP *mlp);

float **batch_output(MLP *mlp, float **input, int batch_size);
float ****batch_forward(MLP *mlp, float **inputs, int batch_size);
void calculate_gradient_and_update(MLP *mlp, float **deltas, float **prev_activations, int batch_size, int layer_idx);
void batch_backward(MLP *mlp, float **inputs, float **targets, int batch_size);

void train(MLP *mlp, float **inputs, float **targets, int num_epochs, int num_samples, int batch_size);
void mnist_predict(MLP *mlp, float *input, int target);
void validate(MLP *mlp, float **inputs, float **targets, int num_samples, int batch_size);
void print_progress(int current_step, int total_steps);
void print_mlp(MLP *mlp);
void check_nan(MLP *mlp);