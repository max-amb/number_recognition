use nalgebra::{DMatrix, DVector};

use crate::neural_network::NN;

pub enum OptimisationAlgorithms {
    StochasticGradientDescent,
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
    pub fn new(
        network: &NN,
        optimisation_algorithm: OptimisationAlgorithms,
        learning_rate: f32,
        cycle_size: usize,
        momentum: Option<f32>
    ) -> Self {
        match optimisation_algorithm {
            OptimisationAlgorithms::StochasticGradientDescent => Optimisation {
                optimisation_algorithm,
                learning_rate,
                cycle_size,
                velocities_weights: None,
                velocities_biases: None,
                momentum: None,
            },
        }
    }

    pub fn calculate_change(
        &mut self,
        delta_weights_sum: &[DMatrix<f32>],
        delta_biases_sum: &[DVector<f32>],
    ) -> (Vec<DMatrix<f32>>, Vec<DVector<f32>>) {
        match &self.optimisation_algorithm {
            OptimisationAlgorithms::StochasticGradientDescent => (
                delta_weights_sum
                    .iter()
                    .map(|x| x * self.learning_rate * (1.0 / (self.cycle_size as f32)))
                    .collect::<Vec<DMatrix<f32>>>(),
                delta_biases_sum
                    .iter()
                    .map(|x| x * self.learning_rate * (1.0 / (self.cycle_size as f32)))
                    .collect::<Vec<DVector<f32>>>(),
            ),
        }
    }
}
