#![allow(unused_imports)]
use std::io;

pub mod cost;
pub mod initialisation;
pub mod layers;
pub mod neural_network;
pub mod tensor;
pub mod training_data;

use cost::CostFunction;
use initialisation::InitialisationOptions;
use nalgebra::DVector;
use neural_network::NN;
use std::sync::mpsc;
use training_data::TrainingData;

use crate::{layers::{Convolution, FullyConnected, Pool, Flatten}, tensor::{Shape, Ten}};
use crate::layers::Layer;

fn main() {
    let data_for_training = TrainingData::new(
        "/home/max/Downloads/train-labels.idx1-ubyte",
        "/home/max/Downloads/train-images.idx3-ubyte",
        60000,
    );
    
    let layers: Vec<Layer> = vec![
        layers::primitives::Layer::CONV(Convolution::new(8, (8, 8), 1, 0, InitialisationOptions::He)),
        layers::primitives::Layer::ACTIVATION(layers::Activation::Relu),
        layers::primitives::Layer::CONV(Convolution::new(8, (2, 2), 1, 0, InitialisationOptions::He)),
        layers::primitives::Layer::ACTIVATION(layers::Activation::Relu),
        layers::primitives::Layer::CONV(Convolution::new(8, (2, 2), 1, 0, InitialisationOptions::He)),
        layers::primitives::Layer::ACTIVATION(layers::Activation::Relu),
        layers::primitives::Layer::FLATTEN(layers::Flatten::new()),
        layers::primitives::Layer::FC(FullyConnected::new(512, InitialisationOptions::He)),
        layers::primitives::Layer::ACTIVATION(layers::Activation::Relu),
        layers::primitives::Layer::FC(FullyConnected::new(256, InitialisationOptions::He)),
        layers::primitives::Layer::ACTIVATION(layers::Activation::Relu),
        layers::primitives::Layer::FC(FullyConnected::new(10, InitialisationOptions::He)),
        layers::primitives::Layer::ACTIVATION(layers::Activation::Relu),
    ];

    let mut network: NN = NN::new(layers, (28, 28).into(), CostFunction::CategoricalCrossEntropy);
    NN::train(&mut network, 512, data_for_training, 0.5, 0.1);
    // let mut network: NN = NN::generate_model_from_file("/home/max/Documents/model.txt").unwrap();
    // let mut network: NN = NN::new(&[784, 512, 256, 10], InitialisationOptions::He, None);

    /*
    network = NN::training(
        network,
        512,
        data_for_training,
        0.99,
        CostFunction::CategoricalCrossEntropy,
        OptimisationAlgorithms::StochasticGradientDescent,
        0.8,
        64,
    );
    NN::output_model_to_file(&network, "/home/max/Documents/model.txt").unwrap();*/
    // input_bmps(&mut network, &CostFunction::CategoricalCrossEntropy);
}

/*
#[allow(dead_code)]
fn input_bmps(network: &mut NN, cost_function: &CostFunction) {
    let (ctrlc_transmitter, ctrlc_reciever) = mpsc::channel();
    ctrlc::set_handler(move || {
        ctrlc_transmitter
            .send(())
            .expect("Could not send signal on channel.")
    })
    .expect("Error setting Ctrl-C handler");

    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).unwrap();
    while ctrlc_reciever.try_recv().is_err() {
        let image = TrainingData::generate_training_data_from_bmp(buffer.trim()).unwrap();
        for i in image.iter().enumerate() {
            if i.0 % 28 == 0 {
                println!()
            };
            print!("{:^3}|", i.1 * 255.0);
        }
        network.layers = NN::forward_pass(network, &image, cost_function);
        println!("\n{:?}", (&network.layers[network.layers.len() - 1]));
        println!("Network classifies number as {:?}", NN::network_classification(&network.layers[network.layers.len() - 1]));

        let prev_buffer: String = buffer;
        buffer = String::new();
        io::stdin().read_line(&mut buffer).unwrap();
        if buffer == "\n" { buffer = prev_buffer; };
    }

    ctrlc_reciever
        .recv()
        .expect("Didn't recieve ctrl-c signal from channel");
    NN::output_model_to_file(
        network,
        "/home/max/projects/number_recognition/models/tmp.txt",
    )
    .unwrap();
}*/
