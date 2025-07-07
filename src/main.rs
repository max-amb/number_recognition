use nalgebra::{DMatrix, DVector, SVector};
use rand::random;
use std::fs;

#[derive(Debug)]
struct NN {
    layers: Vec<DVector<f32>>,
    weights: Vec<DMatrix<f32>>,
    biases: Vec<DVector<f32>>,
}

impl NN {
    fn new (number_of_layers: i32, layer_sizes: &[i32]) -> Result<NN, &'static str> {
        if layer_sizes.len() != number_of_layers as usize {
            return Err("The array of layer sizes has a different number of elements than the number of layers");
        } 

        let layers: Vec<DVector<f32>> = (0..number_of_layers)
            .map(|x| DVector::from_element(layer_sizes[x as usize] as usize, 0.0))
            .collect();
        let weights: Vec<DMatrix<f32>> = (1..number_of_layers)
            .map(|x| DMatrix::from_fn(layer_sizes[x as usize] as usize, layer_sizes[(x-1) as usize] as usize, |_, _| random()))
            .collect();
        let biases: Vec<DVector<f32>> = (1..number_of_layers)
            .map(|x| DVector::from_fn(layer_sizes[x as usize] as usize, |_, _| random()))
            .collect();
        return Ok(Self {
            layers: layers,
            weights: weights,
            biases: biases,
        })
    }

    fn forward_pass (network: &NN) -> Vec<DVector<f32>> {
        let mut new_layers: Vec<DVector<f32>> = vec![ network.layers[0].clone() ];
        for layer in 0..network.weights.len() {
            new_layers.push(&network.weights[layer]*&network.layers[layer+1] + &network.biases[layer]);
        }
        new_layers
    }
}

fn main() {
    let mut network = match NN::new(4, &[718, 16, 16 ]) {
        Ok(network) => network,
        Err(e) => { print!("{e:?}"); panic!("noooo") },
    };
    network.layers = NN::forward_pass(&network);
}
