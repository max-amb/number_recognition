use nalgebra::{DMatrix, DVector};
use crate::neural_network::NN;
use crate::training_data::TrainingData;
use std::io;
use std::sync::{mpsc, Arc};
use std::thread;

pub mod neural_network;
pub mod tests;
pub mod training_data;

const CYCLE_SIZE: usize = 500;

fn main() {
    let mut network = train();

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

    let training_data = Arc::new(TrainingData::new("/home/max/Downloads/train-labels.idx1-ubyte", "/home/max/Downloads/train-images.idx3-ubyte"));
    let mut avg_score: f32 = 0.0;
    let mut iterator_over_cycles = CYCLE_SIZE;

    while avg_score < 0.99 {
        let network_for_this_iter = Arc::new(network.clone());
        let (tx, rx) = mpsc::channel();
        let mut handles = vec![];
        avg_score = 0.0;

        for i in iterator_over_cycles-CYCLE_SIZE..iterator_over_cycles {
            let tx_cloned = tx.clone();
            let new_network = Arc::clone(&network_for_this_iter);
            let training_data_cloned = Arc::clone(&training_data);

            let handle = thread::spawn(move || {
                let new_layers = NN::forward_pass(&new_network, &training_data_cloned.data[i]);
                let costs: DVector<f32> = NN::calculate_cost(&new_layers, &training_data_cloned.labels[i]);
                let (delta_biases, delta_weights) = NN::backpropagation_for_parallelisation(&new_network, costs, &new_layers);

                for i in &delta_biases {
                    if i.iter().find(|x| x.is_nan() ).is_some() { panic!("Got NaN biases") };
                };

                for i in &delta_weights {
                    if i.iter().find(|x| x.is_nan() ).is_some() { panic!("Got NaN weights") };
                };
                
                let mut guessed_correct = false;
                if NN::network_classification(&new_layers[new_layers.len()-1]) == NN::network_classification(&training_data_cloned.labels[i]) { guessed_correct = true };
                tx_cloned.send((delta_biases, delta_weights, guessed_correct)).unwrap()
            });
            handles.push(handle);
        }
        drop(tx);

        for handle in handles {
            let _ = handle.join().unwrap();
        }

        let first_value = rx.recv().unwrap();
        let mut delta_biases_sum = first_value.0;
        let mut delta_weights_sum = first_value.1;
        if first_value.2 { avg_score += 1.0 };
        for recieved in rx {
            if recieved.2 { avg_score += 1.0 };
            delta_biases_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&recieved.0[i]);
            delta_weights_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&recieved.1[i]);
        }

        let to_apply_weights: Vec<DMatrix<f32>> = delta_weights_sum.iter().map(|x| x * 1.0/(avg_score*0.5) * (1.0/(CYCLE_SIZE as f32))).collect();
        let to_apply_biases: Vec<DVector<f32>> = delta_biases_sum.iter().map(|x| x * 1.0/(avg_score*0.5) * (1.0/(CYCLE_SIZE as f32))).collect();
        // let to_apply_weights: Vec<DMatrix<f32>> = delta_weights_sum.iter().map(|x| x * 0.1 * (1.0/(CYCLE_SIZE as f32))).collect();
        // let to_apply_biases: Vec<DVector<f32>> = delta_biases_sum.iter().map(|x| x * 0.1 * (1.0/(CYCLE_SIZE as f32))).collect();


        network.weights.iter_mut().enumerate().for_each(|(i,x)| *x -= &to_apply_weights[i]);
        network.biases.iter_mut().enumerate().for_each(|(i,x)| *x -= &to_apply_biases[i]);

        avg_score = avg_score/(CYCLE_SIZE as f32);
        dbg!(avg_score);
        if iterator_over_cycles + CYCLE_SIZE > training_data.data.len() {
            iterator_over_cycles = CYCLE_SIZE;
        } else {
            iterator_over_cycles += CYCLE_SIZE;
        }
    }
    network
}

#[allow(dead_code)]
fn run_on_testing_data (network: &mut NN) {
    let testing_data = TrainingData::new("/home/max/Downloads/t10k-labels.idx1-ubyte", "/home/max/Downloads/t10k-images.idx3-ubyte");
    let mut correct: usize = 0;
    for j in 0..testing_data.data.len() {
        network.layers = NN::forward_pass(&network, &testing_data.data[j]);
        if NN::network_classification(&network.layers[network.layers.len()-1]) == NN::network_classification(&testing_data.labels[j]) { correct += 1 };
    } 
    dbg!(correct);
}
