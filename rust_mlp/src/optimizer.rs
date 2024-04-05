use crate::mlp::{MultiLayerPerceptron, MLP};
use std::io::{stdout, Write};
use std::marker::PhantomData;
use num_traits::real::Real;

fn print_progress(epoch: usize, current: usize, total: usize) {
    const BAR_LENGTH: usize = 50;

    let progress = (current as f64 / total as f64) * 100.0;
    let progress_bar_length = (progress / 100.0 * BAR_LENGTH as f64) as usize;

    let mut bar = String::from("\rEpoch ");
    bar.push_str(&epoch.to_string());
    bar.push_str(" [");
    bar.push_str(&"=".repeat(progress_bar_length));
    bar.push_str(&" ".repeat(BAR_LENGTH - progress_bar_length));
    bar.push_str("] ");

    let progress_string = format!("{:.2}%", progress);
    bar.push_str(&progress_string);

    print!("{}", bar);
    let _ = stdout().flush();
    if current == total - 1 {
        println!();
    }
}

pub trait Optimizer<T: Real> {
    fn optimize(&mut self, deltas_matrix: &Vec<Vec<T>>, gradients_matrix: &Vec<Vec<Vec<T>>>);
    fn test(&self, input: &Vec<Vec<T>>, target: &Vec<Vec<T>>);
    fn train(&mut self, input: &Vec<Vec<T>>, target: &Vec<Vec<T>>, validation_inputs: &Vec<Vec<T>>, validation_targets: &Vec<Vec<T>>, epochs: usize);
}

pub struct SimpleOptimizer<T: Real> {
    mlp: MultiLayerPerceptron<T>,
    learning_rate: T,
    _t: PhantomData<T>,
}

impl<T: Real> SimpleOptimizer<T> {
    pub fn new(mlp: MultiLayerPerceptron<T>, learning_rate: T) -> Self {
        SimpleOptimizer {
            mlp,
            learning_rate,
            _t: PhantomData,
        }
    }
}

impl<T: Real + std::ops::SubAssign> Optimizer<T> for SimpleOptimizer<T> {
    fn optimize(&mut self, deltas_matrix: &Vec<Vec<T>>, gradients_matrix: &Vec<Vec<Vec<T>>>) {
        for i in 0..self.mlp.layers.len() {
            for j in 0..self.mlp.layers[i].weights.len() {
                for k in 0..self.mlp.layers[i].weights[j].len() {
                    self.mlp.layers[i].weights[j][k] -= self.learning_rate * gradients_matrix[i][j][k];
                }
                self.mlp.layers[i].biases[j] -= self.learning_rate * deltas_matrix[i][j];
            }
        }
    }

    fn test(&self, input: &Vec<Vec<T>>, target: &Vec<Vec<T>>) {
        let mut correct = 0;
        for i in 0..input.len() {
            let forward = self.mlp.forward(&input[i]);
            let max_index = forward[forward.len() - 1].iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
            let target_index = target[i].iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
            if max_index == target_index {
                correct += 1;
            }
        }
        println!("Accuracy: {:?}", correct as f32 / input.len() as f32);
    }

    fn train(&mut self, input: &Vec<Vec<T>>, target: &Vec<Vec<T>>, validation: &Vec<Vec<T>>, validation_targets: &Vec<Vec<T>>, epochs: usize) {
        for epoch in 1..=epochs {
            for i in 0..input.len() {
                let forward = self.mlp.forward(&input[i]);
                let (deltas_matrix, gradients_matrix) = self.mlp.backward(&forward, &input[i], &target[i]);
                self.optimize(&deltas_matrix, &gradients_matrix);
                self.mlp.learning_rate = self.mlp.learning_rate * T::from(0.98).unwrap();
                print_progress(epoch, i, input.len());
            }
            self.test(validation, validation_targets);
        }
    }
}

/*
pub struct AdamOptimizer<T: Real> {
    mlp: MultiLayerPerceptron<T>,
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    first_moment: Vec<Vec<T>>,
    second_moment: Vec<Vec<T>>,
    time_step: T,
    _t: PhantomData<T>,
}

impl<T: Real> AdamOptimizer<T> {
    fn new(mlp: MultiLayerPerceptron<T>, learning_rate: T) -> Self {
        let first_moment = mlp.layers.iter().map(|layer| {
            layer.weights.iter().map(|_| T::zero()).collect()
        }).collect();
        let second_moment = mlp.layers.iter().map(|layer| {
            layer.weights.iter().map(|_| T::zero()).collect()
        }).collect();
        AdamOptimizer {
            mlp,
            learning_rate,
            beta1: T::from(0.9).unwrap(),
            beta2: T::from(0.999).unwrap(),
            epsilon: T::from(1e-8).unwrap(),
            first_moment,
            second_moment,
            time_step: T::zero(),
            _t: PhantomData,
        }
    }
}

impl<T: Real> Optimizer<T> for AdamOptimizer<T> {
    fn optimize(&mut self, deltas_matrix: &Vec<Vec<T>>, gradients_matrix: &Vec<Vec<Vec<T>>>) {
        todo!(); 
    }
    fn test(&self, input: &Vec<Vec<T>>, target: &Vec<Vec<T>>) {
        todo!();
    }
    fn train(&mut self, input: &Vec<Vec<T>>, target: &Vec<Vec<T>>, validation: &Vec<Vec<T>>, validation_targets: &Vec<Vec<T>>, epochs: usize) {
        todo!();
    }
}
*/