use nalgebra::DVector;

use crate::initialisation::InitialisationOptions;
use crate::layers::{Forward, Initialisable};
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
