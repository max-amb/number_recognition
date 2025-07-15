use nalgebra::{DMatrix, DVector};
use training_data::TrainingData;
use crate::neural_network::{NN, InitialisationOptions};
use std::io;

pub mod neural_network;
pub mod tests;
pub mod training_data;

fn main() {
    let data_for_training = TrainingData::new("/home/max/Downloads/train-labels.idx1-ubyte", "/home/max/Downloads/train-images.idx3-ubyte");
    let mut network = match NN::new(4, &[784, 64, 64, 10], InitialisationOptions::default(), None) {
        Ok(network) => network,
        Err(e) => { print!("{e:?}"); panic!("noooo") },
    };
    network = NN::train(network, 5000, data_for_training);
    NN::output_model_to_file(&network, "/home/max/number_recognition/models/x.txt").unwrap();
}

fn input_bmps (network: &mut NN) {
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).unwrap();
    while buffer != "STOP" {
        let image = TrainingData::generate_training_data_from_bmp(&buffer.trim()).unwrap();
        for i in image.iter().enumerate() {
            if i.0 % 28 == 0 { println!("") };
            print!("{:^3}|", i.1*255.0);
        }
        network.layers = NN::forward_pass(&network, &image);
        println!("{}", NN::network_classification(&network.layers[network.layers.len()-1]));

        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
    }
}

#[allow(dead_code)]
fn run_on_testing_data (network: &NN) {
    let testing_data = TrainingData::new("/home/max/Downloads/t10k-labels.idx1-ubyte", "/home/max/Downloads/t10k-images.idx3-ubyte");
    let mut correct: usize = 0;
    for j in 0..testing_data.data.len() {
        let new_layers = NN::forward_pass(&network, &testing_data.data[j]);
        if NN::network_classification(&new_layers[network.layers.len()-1]) == NN::network_classification(&testing_data.labels[j]) { correct += 1 };
    } 
    dbg!(correct);
}
