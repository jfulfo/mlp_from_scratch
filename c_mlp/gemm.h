float **gemm(float **A, float **B, int m, int n, int k);
float **gemm_add(float **A, float **B, int m, int n, int k, float *x);
float **transpose(float **M, int m, int n);
float **allocate_matrix(int m, int n);
void free_matrix(float **M, int m);
void print_matrix(float **M, int m, int n);
void one_hot_vector(float *v, int n);   