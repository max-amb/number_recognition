use nalgebra::DVector;

use crate::initialisation::InitialisationOptions;
use crate::layers::{Forward, Initialisable, Backward};
use crate::tensor::{Shape, Ten};

#[derive(Debug)]
pub struct FullyConnected {
    output_shape: Shape,
    // Need to be options as we do not know shape at creation
    weights: Option<Ten>,
    biases: Option<Ten>,
    initialisation_options: InitialisationOptions,
}

impl FullyConnected {
    fn new(output_shape: Shape, initialisation_options: InitialisationOptions) -> Self {
        Self {
            output_shape,
            weights: None,
            biases: None,
            initialisation_options,
        }
    }
}

impl Forward for FullyConnected {
    fn run(&self, prev_layer: Ten) -> Ten {
        let data: Ten = self.weights.as_ref().unwrap() * prev_layer + self.biases.as_ref().unwrap();
        assert_eq!(data.shape.magnitude(), self.output_shape.magnitude());
        Ten {
            data: data.data,
            shape: self.output_shape,
        }
    }
}

impl Initialisable for FullyConnected {
    fn initialise(&mut self, previous_shape: Shape) -> Shape {
        self.weights = Some(self.initialisation_options.create_matrix(
            (self.output_shape.magnitude(), previous_shape.magnitude(), 1).into(),
            previous_shape.magnitude(),
        ));

        self.biases = Some(DVector::from_element(self.output_shape.magnitude(), 0.0).into());
        self.output_shape
    }
}

pub struct FullyConnectedDelta {
    delta_weights: Ten,
    delta_biases: Ten
}

impl std::ops::Add for FullyConnectedDelta {
    type Output = FullyConnectedDelta; 

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            delta_weights: self.delta_weights + rhs.delta_weights,
            delta_biases: self.delta_biases + rhs.delta_biases
        }
    }
}

impl Backward<FullyConnectedDelta> for FullyConnected {
    fn backprop(&self, following_layer_derivatives: Ten, previous_layer_output: &Ten) -> (Ten, FullyConnectedDelta) {
        let delta_weights = &following_layer_derivatives*(previous_layer_output.transpose());
        let prev_layer_derivatatives= (self.weights.as_ref().unwrap().transpose())*(&following_layer_derivatives);
        (prev_layer_derivatatives, FullyConnectedDelta { delta_weights, delta_biases: following_layer_derivatives })
    }

    fn apply(&mut self, delta: FullyConnectedDelta) {
        self.biases = Some(self.biases.as_ref().unwrap() + delta.delta_biases);
        self.weights = Some(self.weights.as_ref().unwrap() + delta.delta_weights);
    }
}
