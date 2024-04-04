use num_traits::real::Real;
use rand::distributions::{Distribution, Uniform};
use std::marker::PhantomData;

pub trait Initializer<T: Real> {
    fn initialize_weights(&self, n_inputs: usize, n_outputs: usize) -> Vec<Vec<T>>;
    fn initialize_biases(&self, n_outputs: usize) -> Vec<T>;
}

pub struct HeInitializer<T: Real> {
    _t : PhantomData<T>,
}

impl<T: Real> HeInitializer<T> {
    pub fn new() -> Self {
        HeInitializer { _t: PhantomData }
    }
}

impl<T: Real> Initializer<T> for HeInitializer<T> {
    fn initialize_weights(&self, n_inputs: usize, n_outputs: usize) -> Vec<Vec<T>> {
        let mut weights = Vec::with_capacity(n_outputs);
        let uniform = Uniform::new(0.0, 1.0);
        let std_dev = T::sqrt(T::from(2.0).unwrap() / T::from(n_inputs as f32).unwrap());

        for _ in 0..n_outputs {
            let mut weights_row = Vec::with_capacity(n_inputs);
            for _ in 0..n_inputs {
                let rand_val: f32 = uniform.sample(&mut rand::thread_rng());
                let weight = T::from(rand_val).unwrap() * std_dev * T::from(2.0).unwrap() - std_dev;
                weights_row.push(weight);
            }
            weights.push(weights_row);
        }
        weights
    }

    fn initialize_biases(&self, n_outputs: usize) -> Vec<T> {
        vec![T::zero(); n_outputs]
    }
}