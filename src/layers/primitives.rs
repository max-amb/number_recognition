use enum_dispatch::enum_dispatch;
use nalgebra::{DMatrix, DVector, Dyn};

use crate::layers::Activation;
use crate::layers::Convolution;
use crate::layers::FullyConnected;
use crate::layers::Pool;

use crate::tensor::{Shape, Ten};

#[enum_dispatch]
pub trait Forward {
    fn run(&self, prev_layer: Ten) -> Ten;
}

#[enum_dispatch]
pub trait Initialisable {
    fn initialise(&mut self, previous_shape: Shape) -> Shape;
}

#[derive(Debug)]
#[enum_dispatch(Forward, Initialisable)]
pub enum Layer {
    FC(FullyConnected),
    CONV(Convolution),
    POOL(Pool),
    ACTIVATION(Activation),
}
