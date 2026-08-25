use crate::layers::{Backward, Delta, Forward, Initialisable};
use crate::tensor::{Shape, Ten};

/// To flatten input from XD to 1D for operation with fully connected layers
#[derive(Debug)]
pub struct Flatten {
    input_shape: Shape,
}

impl Flatten {
    pub fn new() -> Flatten {
        Flatten { input_shape: (0, 0, 0).into() }
    }
}

impl Forward for Flatten {
    fn run(&self, prev_layer: &Ten) -> Ten {
        Ten { data: prev_layer.data.clone(), shape: (self.input_shape.magnitude(), 1, 1).into() }
    } 
}

impl Backward for Flatten {
    fn apply(&mut self, delta: Delta) {} 

    fn backprop(&self, following_layer_derivatives: Ten, _: &Ten) -> (Ten,Delta) {
        (Ten { data: following_layer_derivatives.data, shape: self.input_shape }, Delta::FLATTEND(()))
    }
}

impl Initialisable for Flatten {
    fn initialise(&mut self, previous_shape: Shape) -> Shape {
        self.input_shape = previous_shape;
        (self.input_shape.magnitude(), 1, 1).into()
    }
}
