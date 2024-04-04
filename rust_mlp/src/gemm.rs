use num_traits::real::Real;

// computes Ax
pub fn matrix_vector_mul<T: Real>(matrix: &Vec<Vec<T>>, vector: &Vec<T>) -> Vec<T> {
    matrix.iter().map(|row| {
        row.iter().zip(vector.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::zero(), |acc, x| acc + x)
    }).collect() 
}

pub fn matrix_matrix_mul<T: Real>(a: &Vec<Vec<T>>, b: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    a.iter().map(|row| {
        b.iter().map(|col| {
            row.iter().zip(col.iter())
                .map(|(&a, &b)| a * b)
                .fold(T::zero(), |acc, x| acc + x)
        }).collect()
    }).collect()
}

pub fn matrix_transpose<T: Real>(matrix: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let n_rows = matrix.len();
    let n_cols = matrix[0].len();
    let mut transposed = Vec::with_capacity(n_cols);
    for i in 0..n_cols {
        let mut row = Vec::with_capacity(n_rows);
        for j in 0..n_rows {
            row.push(matrix[j][i]);
        }
        transposed.push(row);
    }
    transposed
}