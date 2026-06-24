use nalgebra::DVector;

use crate::layers::primitives::{im2col, out_shape};
use crate::layers::{Convolvable, Forward, Mat};

use crate::initialisation::Initialisable;

#[derive(Debug)]
pub struct Pool {
    shape: (usize, usize),
    stride: usize,
    pooling_function: fn(DVector<f32>) -> f32,
}

impl Pool {
    fn new(shape: (usize, usize), stride: usize, pooling_function: fn(DVector<f32>) -> f32) -> Self {
        Self { shape, stride, pooling_function }
    } 
}

impl Initialisable for Pool {
    fn initialise(self, _previous_shape: (usize, usize)) -> Self {
        self     
    } 
}

impl Convolvable for Pool {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn zero_padding(&self) -> usize {
        0
    }

    fn stride(&self) -> usize {
        self.stride
    }
}

impl Forward for Pool {
    fn run(&self, prev_layer: Mat) -> Mat {
        let shape = out_shape(&prev_layer, self);
        let prev_columnised = im2col(prev_layer, self);
        let mut result: Vec<f32> = Vec::with_capacity(prev_columnised.ncols());
        for col in prev_columnised.column_iter() {
            result.push((self.pooling_function)(col.into()));
        }
        Mat {
            data: DVector::from_vec(result),
            shape,
        }
    }
}
