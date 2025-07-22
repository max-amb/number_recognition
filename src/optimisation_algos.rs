use nalgebra::{DVector, DMatrix};

use crate::neural_network::NN;

pub enum OptimisationAlgorithms {
    StochasticGradientDescent,
    ClassicMomentum,
}

pub struct Optimisation {
    optimisation_algorithm: OptimisationAlgorithms,
    learning_rate: f32,
    cycle_size: usize,
    velocities_weights: Option<Vec<DMatrix<f32>>>,
    velocities_biases: Option<Vec<DVector<f32>>>,
    momentum: Option<f32>,
}

impl Optimisation {
    pub fn new (network: &NN, optimisation_algorithm: OptimisationAlgorithms, learning_rate: f32, cycle_size: usize, momentum: Option<f32>) -> Self {
        match optimisation_algorithm {
            OptimisationAlgorithms::StochasticGradientDescent =>
                Optimisation { optimisation_algorithm, learning_rate, cycle_size, velocities_weights: None, velocities_biases: None, momentum: None },
            OptimisationAlgorithms::ClassicMomentum => 
                Optimisation { optimisation_algorithm,
                    learning_rate,
                    cycle_size,
                    velocities_weights: Some((1..network.layers.len())
                        .map(|x| DMatrix::from_element(network.layers[x].nrows(), network.layers[x-1].nrows(), 0.0))
                        .collect()),
                    velocities_biases: Some((1..network.layers.len())
                        .map(|x| DVector::from_element(network.layers[x].nrows(), 0.0))
                        .collect()),
                    momentum: Some(momentum.expect("Momentum required for classic momentum optimisation algorithm"))
                }
        }
    }

    pub fn calculate_change (&mut self, delta_weights_sum: Vec<DMatrix<f32>>, delta_biases_sum: Vec<DVector<f32>>, ) -> (Vec<DMatrix<f32>>, Vec<DVector<f32>>) {
        match &self.optimisation_algorithm {
            OptimisationAlgorithms::StochasticGradientDescent => {
                (delta_weights_sum.iter().map(|x| x * self.learning_rate * (1.0/(self.cycle_size as f32))).collect::<Vec<DMatrix<f32>>>(),
                delta_biases_sum.iter().map(|x| x * self.learning_rate * (1.0/(self.cycle_size as f32))).collect::<Vec<DVector<f32>>>())
            },
            OptimisationAlgorithms::ClassicMomentum => {
                let to_apply_weights: Vec<DMatrix<f32>> = delta_weights_sum.iter().map(|x| x * self.learning_rate * (1.0/(self.cycle_size as f32))).collect();
                let to_apply_biases: Vec<DVector<f32>> = delta_biases_sum.iter().map(|x| x * self.learning_rate * (1.0/(self.cycle_size as f32))).collect();
                self.velocities_biases = Some(self.velocities_biases.as_mut().unwrap().iter().enumerate().map(|(i,x)| (x * self.momentum.unwrap()) + &to_apply_biases[i]).collect());
                self.velocities_weights = Some(self.velocities_weights.as_mut().unwrap().iter().enumerate().map(|(i,x)| (x * self.momentum.unwrap()) + &to_apply_weights[i]).collect());
                (self.velocities_weights.clone().unwrap(), self.velocities_biases.clone().unwrap())
            },
        }
    }
}
