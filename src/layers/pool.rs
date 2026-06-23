use nalgebra::DMatrix;

use crate::layers::{Forward, Mat};

#[derive(Debug)]
pub struct Pool {
    size: u64,
    stride: u64,
    pooling_function: fn(DMatrix<f32>) -> DMatrix<f32>
}

impl Forward for Pool {
    fn run(&self, prev_layer: Mat) -> Mat {
         
     } 
}
