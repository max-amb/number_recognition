use nalgebra::{DMatrix, DVector};

use crate::layers::{Forward, Mat};

#[derive(Debug)]
pub struct FullyConnected {
    output_shape: (usize, usize),
    weights: DMatrix<f32>,
    biases: DVector<f32>,
}

impl Forward for FullyConnected {
    fn run(&self, prev_layer: Mat) -> Mat {
        let data = &self.weights * prev_layer.data + &self.biases;
        assert_eq!(data.nrows(), self.output_shape.0 * self.output_shape.1);
        Mat {
            data,
            shape: self.output_shape,
        }
    }
}
