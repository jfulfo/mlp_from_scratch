// abstract implementation of mlp, with layers and activation functions
// concretization is meant to be done in main

use crate::loss::LossFunction;
use crate::layer::{DenseLayer, Layer};
use num_traits::real::Real;

pub trait MLP<T: Real> {
    fn forward(&self, input: &Vec<T>) -> Vec<Vec<T>>;
    fn backward(&mut self, input: &Vec<Vec<T>>, input: &Vec<T>, target: &Vec<T>) -> (Vec<Vec<T>>, Vec<Vec<Vec<T>>>);
}

pub struct MultiLayerPerceptron<T: Real> {
    pub layers: Vec<DenseLayer<T>>,
    pub loss: Box<dyn LossFunction<T>>,
}

impl<T: Real> MultiLayerPerceptron<T> {
    pub fn new(loss: impl LossFunction<T> + 'static) -> Self {
        MultiLayerPerceptron {
            layers: Vec::new(),
            loss: Box::new(loss),
        }
    }

    pub fn add_layer(&mut self, layer: DenseLayer<T>) {
        self.layers.push(layer);
    }
}

impl<T: Real> MLP<T> for MultiLayerPerceptron<T> {
    fn forward(&self, input: &Vec<T>) -> Vec<Vec<T>> {
        let mut layer_outputs: Vec<Vec<T>> = Vec::new();
        let mut current_input = input.clone();
        for layer in &self.layers {
            let output = layer.forward(&current_input);
            layer_outputs.push(output.clone());
            current_input = output;
        }
        layer_outputs
    }

    fn backward(&mut self, layer_outputs: &Vec<Vec<T>>, input: &Vec<T>, target: &Vec<T>) -> (Vec<Vec<T>>, Vec<Vec<Vec<T>>>) {
        // first, calculate deltas for the last layer
        let mut deltas_matrix: Vec<Vec<T>> = Vec::new();
        let mut gradients_matrix: Vec<Vec<Vec<T>>> = Vec::new();
        let mut deltas: Vec<T> = self.loss.derivative(&layer_outputs[layer_outputs.len() - 1], &target);
        let mut gradients: Vec<Vec<T>>;
        // then, calculate deltas for the other layers
        for i in (0..self.layers.len()).rev() {
            let layer_input = if i == 0 {
                input.clone()
            } else {
                layer_outputs[i - 1].clone()
            };
            let next_weights = if i == self.layers.len() - 1 {
                Vec::new()
            } else {
                self.layers[i + 1].weights.clone()
            };
            (deltas, gradients) = self.layers[i].backward(&layer_input, &layer_outputs[i], &next_weights, &deltas);
            deltas_matrix.push(deltas.clone());
            gradients_matrix.push(gradients);

        }
        deltas_matrix.reverse();
        gradients_matrix.reverse();
        (deltas_matrix, gradients_matrix) // return the deltas and gradients for each layer to the optimizer
    }
}