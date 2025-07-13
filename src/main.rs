use nalgebra::{DMatrix, DVector};
use crate::neural_network::NN;
use crate::training_data::TrainingData;
use std::io;
use std::sync::mpsc;
use std::thread;

pub mod neural_network;
pub mod tests;
pub mod training_data;

const CYCLE_SIZE: usize = 500;

fn main() {

    let mut network = train();

    /*
    let testing_data = TrainingData::new("/home/max/Downloads/t10k-labels.idx1-ubyte", "/home/max/Downloads/t10k-images.idx3-ubyte");
    let mut correct: usize = 0;
    for j in 0..testing_data.data.len() {
        network.layers = NN::forward_pass(&network, &training_data.data[j]);
        if NN::network_classification(&network.layers[network.layers.len()-1]) == NN::network_classification(&training_data.labels[j]) { correct += 1 };
    } 
    dbg!(correct); */

    println!("Now enter file paths of bmps for analysing, STOP to end");
    let mut buffer = String::new();
    io::stdin().read_line(&mut buffer).unwrap();
    while buffer != "STOP" {
        network.layers = NN::forward_pass(&network, &TrainingData::generate_training_data_from_bmp(&buffer.trim()).unwrap());
        dbg!(NN::network_classification(&network.layers[network.layers.len()-1]));

        io::stdin().read_line(&mut buffer).unwrap();
    }
}

fn train () -> NN { 
    let mut network = match NN::new(4, &[784, 16, 16, 10]) {
        Ok(network) => network,
        Err(e) => { print!("{e:?}"); panic!("noooo") },
    };

    let training_data = TrainingData::new("/home/max/Downloads/train-labels.idx1-ubyte", "/home/max/Downloads/train-images.idx3-ubyte");
    let mut avg_cost: f32 = 0.0;
    let mut j = 0;
    let mut n_cycles: u32 = 0;
    while avg_cost < 0.99 {
        n_cycles += 1;
        j += CYCLE_SIZE;
        if j+CYCLE_SIZE >= training_data.data.len() { j = 1 };

        avg_cost = 0.0;
        network.layers = NN::forward_pass(&network, &training_data.data[j]);
        let costs: DVector<f32> = NN::calculate_cost(&network, &training_data.labels[j]);
        let (mut delta_biases_sum, mut delta_weights_sum) = NN::backpropagation(&network, costs);
        for i in j+1..(j+CYCLE_SIZE) {
            network.layers = NN::forward_pass(&network, &training_data.data[i]);
            let costs: DVector<f32> = NN::calculate_cost(&network, &training_data.labels[i]);
            let (delta_biases, delta_weights) = NN::backpropagation(&network, costs);

            for i in &delta_biases {
                if i.iter().find(|x| x.is_nan() ).is_some() { panic!("Got NaN biases") };
            };

            for i in &delta_weights {
                if i.iter().find(|x| x.is_nan() ).is_some() { panic!("Got NaN weights") };
            };

            delta_biases_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&delta_biases[i]);
            delta_weights_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&delta_weights[i]);


            if NN::network_classification(&network.layers[network.layers.len()-1]) == NN::network_classification(&training_data.labels[i]) { avg_cost += 1.0 };
        }

        
        let to_apply_weights: Vec<DMatrix<f32>> = delta_weights_sum.iter().map(|x| x * 1.0/(avg_cost*0.5) * (1.0/(CYCLE_SIZE as f32))).collect();
        let to_apply_biases: Vec<DVector<f32>> = delta_biases_sum.iter().map(|x| x * 1.0/(avg_cost*0.5) * (1.0/(CYCLE_SIZE as f32))).collect();
        // let to_apply_weights: Vec<DMatrix<f32>> = delta_weights_sum.iter().map(|x| x * 0.1 * (1.0/(CYCLE_SIZE as f32))).collect();
        // let to_apply_biases: Vec<DVector<f32>> = delta_biases_sum.iter().map(|x| x * 0.1 * (1.0/(CYCLE_SIZE as f32))).collect();


        network.weights.iter_mut().enumerate().for_each(|(i,x)| *x -= &to_apply_weights[i]);
        network.biases.iter_mut().enumerate().for_each(|(i,x)| *x -= &to_apply_biases[i]);
        avg_cost = avg_cost/(CYCLE_SIZE as f32);

        if n_cycles % 1000 == 0 {
            dbg!(avg_cost);
        };
    }
    network
}
