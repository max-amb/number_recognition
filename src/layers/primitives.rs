use enum_dispatch::enum_dispatch;

use crate::layers::{Activation, Flatten};
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
    fn run(&self, prev_layer: &Ten) -> Ten;
}

#[enum_dispatch]
pub trait Backward {
    fn backprop(&self, following_layer_derivatives: Ten, previous_layer_output: &Ten) -> (Ten, Delta);
    fn apply(&mut self, delta: Delta);
}

#[derive(Debug)]
pub enum Delta {
    FCD(FullyConnectedDelta),
    CONVD(ConvolutionDelta),
    POOLD(()),
    ACTIVATIOND(()),
    FLATTEND(())
}

impl std::ops::Mul<f32> for Delta {
    type Output = Delta;

    fn mul(self, rhs: f32) -> Self::Output {
        match self {
            Self::FCD(fc) => {
                return Self::FCD(fc * rhs);
            } 

            Self::CONVD(conv) => {
                return Self::CONVD(conv * rhs);
            }

            Self::POOLD(_) => {
                return Self::POOLD(());
            }

            Self::ACTIVATIOND(_) => {
                return Self::ACTIVATIOND(());
            }

            Self::FLATTEND(_) => {
                return Self::FLATTEND(());
            }
        }
    }
}

impl std::ops::Add for Delta {
    type Output = Delta; 
    fn add(self, rhs: Self) -> Self::Output {
        // TODO: Complete
        match self {
            Self::FCD(fc) => {
                if let Self::FCD(rhs_fc) = rhs {
                    return Delta::FCD(fc.add(rhs_fc))
                } else {
                    panic!();
                }
            } 

            Self::CONVD(conv) => {
                if let Self::CONVD(rhs_conv) = rhs {
                    return Delta::CONVD(conv.add(rhs_conv));
                } else {
                    panic!();
                }
            }

            Self::POOLD(_) => {
                if let Self::POOLD(_) = rhs {
                    return Delta::POOLD(());
                } else {
                    panic!();
                }
            }

            Self::ACTIVATIOND(_) => {
                if let Self::ACTIVATIOND(_) = rhs {
                    return Delta::ACTIVATIOND(());
                } else {
                    panic!();
                }
            }

            Self::FLATTEND(_) => {
                if let Self::FLATTEND(_) = rhs {
                    return Delta::FLATTEND(());
                } else {
                    panic!();
                }
            }
        }
    }
}

#[derive(Debug)]
#[enum_dispatch(Forward, Initialisable, Backward)]
pub enum Layer {
    FC(FullyConnected),
    CONV(Convolution),
    POOL(Pool),
    ACTIVATION(Activation),
    FLATTEN(Flatten)
}
