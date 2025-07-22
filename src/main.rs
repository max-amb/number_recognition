use std::io;

pub mod optimisation_algos;
pub mod neural_network;
pub mod tests;
pub mod training_data;

use training_data::TrainingData;
use neural_network::{NN, InitialisationOptions, CostFunction};
use optimisation_algos::OptimisationAlgorithms;

fn main() {
    let data_for_training = TrainingData::new("/home/max/Downloads/train-labels.idx1-ubyte", "/home/max/Downloads/train-images.idx3-ubyte");
    let mut network: NN = match NN::new(4, &[784, 16, 16, 10], InitialisationOptions::He, None) {
        Ok(network) => network,
        Err(e) => { print!("{e:?}"); panic!("noooo") },
    };

    network = NN::training(network, 500, data_for_training, 0.999, CostFunction::CrossEntropy, OptimisationAlgorithms::StochasticGradientDescent);
    input_bmps(&mut network, &CostFunction::Quadratic);

    // Handling ctrl-c gracefully
    /*
    let (ctrlc_transmitter, ctrlc_reciever) = mpsc::channel();
    ctrlc::set_handler(move || ctrlc_transmitter.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");
    ctrlc_reciever.recv().expect("Didn't recieve ctrl-c signal from channel");
    NN::output_model_to_file(&network, "/home/max/number_recognition/models/tmp.txt").unwrap(); */
}

#[allow(dead_code)]
fn input_bmps (network: &mut NN, cost_function: &CostFunction) {
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).unwrap();
    while buffer != "STOP" {
        let image = TrainingData::generate_training_data_from_bmp(buffer.trim()).unwrap();
        for i in image.iter().enumerate() {
            if i.0 % 28 == 0 { println!() };
            print!("{:^3}|", i.1*255.0);
        }
        network.layers = NN::forward_pass(network, &image, cost_function);
        println!("\n{:?}", (&network.layers[network.layers.len()-1]));

        buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
    }
}

#[allow(dead_code)]
fn run_on_testing_data (network: &NN, cost_function: &CostFunction) {
    let testing_data = TrainingData::new("/home/max/Downloads/t10k-labels.idx1-ubyte", "/home/max/Downloads/t10k-images.idx3-ubyte");
    let mut correct: usize = 0;
    for j in 0..testing_data.data.len() {
        let new_layers = NN::forward_pass(network, &testing_data.data[j], cost_function);
        if NN::network_classification(&new_layers[network.layers.len()-1]) == NN::network_classification(&testing_data.labels[j]) { correct += 1 };
    } 
    dbg!(correct);
}
