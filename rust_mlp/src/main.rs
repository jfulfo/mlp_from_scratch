mod optimizer;
mod initializer;
mod layer;
mod mlp;
mod loss;
mod gemm;

use optimizer::{SimpleOptimizer, Optimizer};
use layer::{ReLU, Softmax, DenseLayer};
use loss::CrossEntropyLoss;
use mlp::MultiLayerPerceptron;
use csv;


// first column is the target, the rest are the input
fn read_mnist_csv(file_path: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rdr = csv::Reader::from_path(file_path).expect("Error reading file");
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        let target = record[0].parse::<f32>().unwrap();
        let input: Vec<f32> = record.iter().skip(1).map(|x| x.parse::<f32>().unwrap() / 255.0).collect();
        inputs.push(input);
        targets.push((0..10).map(|x| if x == target as usize { 1.0 } else { 0.0 }).collect());
    }
    (inputs, targets)
}

fn main() {
    let mnist_train_csv = "data/mnist_train.csv";
    let mnist_test_csv = "data/mnist_test.csv";

    println!("Reading MNIST data...");
    let (mut train_inputs, mut train_targets) = read_mnist_csv(mnist_train_csv);
    // split train_inputs and train_targets into train and validation
    let validation_inputs = train_inputs[0..1000].to_vec();
    let validation_targets = train_targets[0..1000].to_vec();
    train_inputs = train_inputs[1000..].to_vec();
    train_targets = train_targets[1000..].to_vec();

    // create the MLP
    let learning_rate = 0.005;
    let initializer = initializer::HeInitializer::<f32>::new();
    let mut mnist_mlp = MultiLayerPerceptron::<f32>::new(CrossEntropyLoss::new());
    mnist_mlp.add_layer(DenseLayer::new(784, 128, &initializer, ReLU::<f32>::new()));
    mnist_mlp.add_layer(DenseLayer::new(128, 64, &initializer, ReLU::<f32>::new()));
    mnist_mlp.add_layer(DenseLayer::new(64, 32, &initializer, ReLU::<f32>::new()));
    mnist_mlp.add_layer(DenseLayer::new(32, 10, &initializer, Softmax::<f32>::new()));

    println!("Training MLP...");
    let epochs = 3;
    let mut optimizer = SimpleOptimizer::new(mnist_mlp, learning_rate);
    optimizer.train(&train_inputs, &train_targets, &validation_inputs, &validation_targets, epochs);

    println!("Testing MLP...");
    let (test_inputs, test_targets) = read_mnist_csv(mnist_test_csv);
    optimizer.test(&test_inputs, &test_targets);
}
