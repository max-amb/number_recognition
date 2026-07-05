use enum_dispatch::enum_dispatch;
use std::ops::Add;

use crate::layers::Activation;
use crate::layers::Convolution;
use crate::layers::FullyConnected;
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

pub trait Backward<D: Add<Output = D>> {
    // TODO: Make more elegant input (perhaps ref to some network struct?)
    fn backprop(&self, following_layer_derivatives: Ten, previous_layer_output: &Ten) -> (Ten, D);
    fn apply(&mut self, delta: D);
}

#[derive(Debug)]
#[enum_dispatch(Forward, Initialisable)]
pub enum Layer {
    FC(FullyConnected),
    CONV(Convolution),
    POOL(Pool),
    ACTIVATION(Activation),
}
