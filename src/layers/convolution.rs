use nalgebra::{DMatrix};

use crate::layers::{Forward, Matrix};

#[derive(Debug)]
pub struct Kernel {
    kernel: DMatrix<f32>,
    weights: DMatrix<f32>,
    biases: DMatrix<f32>,
    stride: u64,
    zero_padding: u64
}

#[derive(Debug)]
pub struct Convolution {
    filters: Vec<Kernel>
}

impl Forward for Convolution {
    fn run(&self, prev_layer: Matrix) -> Matrix {
         todo!();
    } 
}
