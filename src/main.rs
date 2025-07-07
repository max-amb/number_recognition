use nalgebra::{DMatrix, DVector};
use rand::random;

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
}

fn main() {
    let mut network = match NN::new(4, &[718, 16, 16 ]) {
        Ok(network) => network,
        Err(e) => { print!("{e:?}"); panic!("noooo") },
    };
}
