use num_traits::real::Real;
use std::marker::PhantomData;

pub trait LossFunction<T: Real> {
    fn loss(&self, output: &Vec<T>, target: &Vec<T>) -> T;
    fn derivative(&self, output: &Vec<T>, target: &Vec<T>) -> Vec<T>;
}

pub struct CrossEntropyLoss<T: Real> {
    _t: PhantomData<T>,
}

impl<T: Real> CrossEntropyLoss<T> {
    pub fn new() -> Self {
        CrossEntropyLoss { _t: PhantomData }
    }
}

impl<T: Real> LossFunction<T> for CrossEntropyLoss<T> {
    fn loss(&self, output: &Vec<T>, target: &Vec<T>) -> T {
        target.iter().zip(output.iter())
            .map(|(t, o)| -*t * o.ln())
            .fold(T::zero(), |acc, x| acc + x)
    }

    fn derivative(&self, output: &Vec<T>, target: &Vec<T>) -> Vec<T> {
        output.iter().zip(target.iter())
            .map(|(o, t)| *o - *t)
            .collect()
    }
}