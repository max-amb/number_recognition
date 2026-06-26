use nalgebra::DVector;

use crate::layers::{Forward, Initialisable};
use crate::tensor::{Ten, Shape};

#[derive(Debug)]
pub enum Activation {
    Sigmoid,
    Softmax,
    LeakyRelu { alpha: f32 },
    Relu,
}

impl Forward for Activation {
    fn run(&self, prev_layer: Ten) -> Ten {
        prev_layer.fmap(|val| match self {
            Activation::Sigmoid => sigmoid(val),
            Activation::Relu => relu(val),
            Activation::LeakyRelu { alpha } => leaky_relu(val, *alpha),
            Activation::Softmax => softmax(val),
        })
    }
}

impl Initialisable for Activation {
    fn initialise(&mut self, previous_shape: Shape) -> Shape {
        previous_shape
    }
}

fn sigmoid(layer: DVector<f32>) -> DVector<f32> {
    layer.map(|x| 1.0 / (1.0 + (-x).exp()))
}

fn softmax(layer: DVector<f32>) -> DVector<f32> {
    let layers_max = layer.max();
    let exponentials =
        DVector::from_iterator(layer.nrows(), layer.iter().map(|x| (x - layers_max).exp()));
    &exponentials / exponentials.sum()
}

fn leaky_relu(layer: DVector<f32>, alpha: f32) -> DVector<f32> {
    layer.map(|x| if x.lt(&0.0) { alpha * x } else { x })
}

fn relu(layer: DVector<f32>) -> DVector<f32> {
    layer.map(|x| if x.lt(&0.0) { 0.0 } else { x })
}

pub fn leaky_relu_derivative(input: f32, alpha: f32) -> f32 {
    if input.lt(&0.0) { alpha } else { 1.0 }
}

pub fn relu_derivative(input: f32) -> f32 {
    if input.lt(&0.0) { 0.0 } else { 1.0 }
}
