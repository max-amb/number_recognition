use enum_dispatch::enum_dispatch;

use crate::layers::Activation;
use crate::layers::{Convolution, ConvolutionDelta};
use crate::layers::{FullyConnected, FullyConnectedDelta};
use crate::layers::Pool;

use crate::tensor::{Shape, Ten};

#[enum_dispatch]
pub trait Initialisable {
    fn initialise(&mut self, previous_shape: Shape) -> Shape;
}

#[enum_dispatch]
pub trait Forward {
    fn run(&self, prev_layer: Ten) -> Ten;
}

#[enum_dispatch]
pub trait Backward {
    fn backprop(&self, following_layer_derivatives: Ten, previous_layer_output: &Ten) -> (Ten, Delta);
    fn apply(&mut self, delta: Delta);
}

pub enum Delta {
    FCD(FullyConnectedDelta),
    CONVD(ConvolutionDelta),
    POOLD(()),
    ACTIVATIOND(())
}

#[derive(Debug)]
#[enum_dispatch(Forward, Initialisable, Backward)]
pub enum Layer {
    FC(FullyConnected),
    CONV(Convolution),
    POOL(Pool),
    ACTIVATION(Activation),
}
