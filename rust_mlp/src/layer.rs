use crate::initializer::Initializer;
use crate::gemm::*;
use std::marker::PhantomData;
use num_traits::real::Real;

pub trait ActivationFunction<T: Real> {
    fn activate_layer(&self, input: &Vec<T>) -> Vec<T> {
        input.iter().map(|x| self.activate(*x)).collect()
    }
    fn activate_batch(&self, input: Vec<Vec<T>>) -> Vec<Vec<T>> {
        input.iter().map(|row| self.activate_layer(&row)).collect()
    }
    fn activate(&self, x: T) -> T;
    fn derivative(&self, x: T) -> T;
}

pub struct ReLU<T: Real> {
    _t: PhantomData<T>,
}

impl<T: Real> ReLU<T> {
    pub fn new() -> Self {
        ReLU { _t: PhantomData }
    }
}

impl<T: Real> ActivationFunction<T> for ReLU<T> {
    fn activate(&self, x: T) -> T {
        if x > T::zero() {
            x
        } else {
            T::zero()
        }
    }

    fn derivative(&self, x: T) -> T {
        if x > T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }
}


pub struct Softmax<T: Real> {
    _t: PhantomData<T>,
}

impl<T: Real> Softmax<T> {
    pub fn new() -> Self {
        Softmax { _t: PhantomData }
    }
}

impl<T: Real + std::iter::Sum> ActivationFunction<T> for Softmax<T> {
    fn activate_layer(&self, input: &Vec<T>) -> Vec<T> {
        let max_val = input.iter().fold(T::min_value(), |acc, &x| acc.max(x));
        let exp_sum: T = input.iter().map(|x| (*x - max_val).exp()).sum();
        input.iter().map(|x| (*x - max_val).exp() / exp_sum).collect()
    }

    fn activate(&self, x: T) -> T {
        x
    }

    fn derivative(&self, x: T) -> T {
        x
    }
}

pub trait Layer<T: Real> {
    fn forward(&self, input: &Vec<T>) -> Vec<T>;
    fn batch_forward(&self, input: &Vec<Vec<T>>) -> Vec<Vec<T>>;
    fn backward_last_layer(&mut self, input: &Vec<T>, output: &Vec<T>, next_deltas: &Vec<T>) -> (Vec<T>, Vec<Vec<T>>);
    fn backward(&mut self, input: &Vec<T>, output: &Vec<T>, next_weights: &Vec<Vec<T>>, next_deltas: &Vec<T>) -> (Vec<T>, Vec<Vec<T>>);
}

pub struct DenseLayer<T: Real> {
    pub weights: Vec<Vec<T>>,
    pub biases: Vec<T>,
    pub activation: Box<dyn ActivationFunction<T>>,
}

impl<T: Real> DenseLayer<T> {
    pub fn new(n_inputs: usize, n_outputs: usize, initializer: &dyn Initializer<T>, activation: impl ActivationFunction<T> + 'static) -> Self {
        let weights = initializer.initialize_weights(n_inputs, n_outputs);
        let biases = initializer.initialize_biases(n_outputs);
        DenseLayer { weights, biases, activation: Box::new(activation) }
    }
}

impl<T: Real> Layer<T> for DenseLayer<T> {
    fn forward(&self, input: &Vec<T>) -> Vec<T> {
        // compute weighted sum of inputs
        let output = matrix_vector_mul(&self.weights, &input);

        // apply activation function
        self.activation.activate_layer(&output)
    }

    fn batch_forward(&self, input: &Vec<Vec<T>>) -> Vec<Vec<T>> {
        let output = matrix_matrix_mul(&self.weights, &input);

        self.activation.activate_batch(output)
    }

    fn backward_last_layer(&mut self, input: &Vec<T>, output: &Vec<T>, next_deltas: &Vec<T>) -> (Vec<T>, Vec<Vec<T>>) {
        let mut deltas = vec![T::zero(); self.biases.len()];
        let mut gradients = vec![vec![T::zero(); self.weights[0].len()]; self.weights.len()];

        for (i, delta) in deltas.iter_mut().enumerate() {
            *delta = next_deltas[i] * self.activation.derivative(output[i]);
        }

        for (i, delta) in deltas.iter().enumerate() {
            for (j, &input_val) in input.iter().enumerate() {
                gradients[i][j] = *delta * input_val;
            }
        }

        (deltas, gradients)
    }

    fn backward(&mut self, input: &Vec<T>, output: &Vec<T>, next_weights: &Vec<Vec<T>>, next_deltas: &Vec<T>) -> (Vec<T>, Vec<Vec<T>>) {
        if next_weights.is_empty() {
            return self.backward_last_layer(input, output, next_deltas);
        }

        // calculate deltas using chain rule
        let transpose_weights = matrix_transpose(&next_weights);
        let mut deltas = matrix_vector_mul(&transpose_weights, &next_deltas);
        let mut gradients = vec![vec![T::zero(); self.weights[0].len()]; self.weights.len()];

        // multiply deltas by derivative of activation function
        for (i, delta) in deltas.iter_mut().enumerate() {
            *delta = *delta * self.activation.derivative(output[i]);
        }

        // calculate gradients for weights
        for (i, delta) in deltas.iter().enumerate() {
            for (j, &input_val) in input.iter().enumerate() {
                gradients[i][j] = *delta * input_val;
            }
        }

        // clip gradients
        for i in 0..gradients.len() {
            for j in 0..gradients[i].len() {
                if gradients[i][j].abs() > T::from(5.0).unwrap() {
                    gradients[i][j] = T::from(5.0).unwrap();
                }
                if gradients[i][j].abs() < T::from(1e-5).unwrap() {
                    gradients[i][j] = T::from(1e-5).unwrap();
                }
            }
        }

        (deltas, gradients)
    }
}
