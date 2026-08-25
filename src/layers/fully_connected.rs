use nalgebra::DVector;

use crate::initialisation::InitialisationOptions;
use crate::layers::primitives::Delta;
use crate::layers::{Backward, Forward, Initialisable};
use crate::tensor::{Shape, Ten};

#[derive(Debug)]
pub struct FullyConnected {
    output_size: usize,
    // Need to be options as we do not know shape at creation
    weights: Option<Ten>,
    biases: Option<Ten>,
    initialisation_options: InitialisationOptions,
}

impl FullyConnected {
    pub fn new(output_size: usize, initialisation_options: InitialisationOptions) -> Self {
        Self {
            output_size,
            weights: None,
            biases: None,
            initialisation_options,
        }
    }
}

impl Forward for FullyConnected {
    fn run(&self, prev_layer: &Ten) -> Ten {
        let data: Ten = self.weights.as_ref().unwrap() * prev_layer + self.biases.as_ref().unwrap();
        Ten {
            data: data.data,
            shape: (self.output_size, 1, 1).into()
        }
    }
}

impl Initialisable for FullyConnected {
    fn initialise(&mut self, previous_shape: Shape) -> Shape {
        self.weights = Some(self.initialisation_options.create_matrix(
            (self.output_size, previous_shape.magnitude(), 1).into(),
            previous_shape.magnitude(),
        ));

        self.biases = Some(DVector::from_element(self.output_size, 0.0).into());
        (self.output_size, 1, 1).into()
    }
}

#[derive(Debug)]
pub struct FullyConnectedDelta {
    delta_weights: Ten,
    delta_biases: Ten,
}

impl std::ops::Mul<f32> for FullyConnectedDelta {
    type Output = FullyConnectedDelta;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            delta_weights: self.delta_weights * rhs,
            delta_biases: self.delta_biases * rhs,
        }
    }
}


impl std::ops::Add for FullyConnectedDelta {
    type Output = FullyConnectedDelta;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            delta_weights: self.delta_weights + rhs.delta_weights,
            delta_biases: self.delta_biases + rhs.delta_biases,
        }
    }
}

/// When we are in fully connected layers, (following_layer_derivatives) is \delta from above layer
impl Backward for FullyConnected {
    fn backprop(
        &self,
        delta: Ten,
        previous_layer_output: &Ten,
    ) -> (Ten, Delta) {
        let delta_weights = &delta * (previous_layer_output.transpose());
        (
            self.weights.as_ref().unwrap().transpose() * &delta,
            Delta::FCD(FullyConnectedDelta {
                delta_weights,
                delta_biases: delta,
            }),
        )
    }

    fn apply(&mut self, delta: Delta) {
        if let Delta::FCD(delta) = delta {
            self.biases = Some(self.biases.as_ref().unwrap() - delta.delta_biases);
            self.weights = Some(self.weights.as_ref().unwrap() - delta.delta_weights);
        } else {
            panic!()
        }
    }
}
