pub mod activations;
pub mod convolution;
pub mod convolvable;
pub mod fully_connected;
pub mod pool;
pub mod primitives;

pub use activations::Activation;
pub use convolution::{Convolution, Kernel, ConvolutionDelta};
pub use convolvable::Convolvable;
pub use fully_connected::{FullyConnected, FullyConnectedDelta};
pub use pool::{Pool};
pub use primitives::{Backward, Forward, Initialisable, Delta, Layer};
