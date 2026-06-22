use nalgebra::{DMatrix, DVector};

use crate::layers::{Forward, Matrix};

#[derive(Debug)]
pub struct FullyConnected {
    output_size: usize,
    weights: DMatrix<f32>,
    biases: DVector<f32>
}

impl Forward for FullyConnected {
    fn run(&self, prev_layer: Matrix) -> Matrix {
        Matrix {
            data: &self.weights * prev_layer.data + &self.biases,
            shape: (self.output_size, 1)
        }
    } 
}
