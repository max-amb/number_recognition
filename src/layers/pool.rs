use nalgebra::DVector;

use crate::layers::primitives::{im2col, out_shape};
use crate::layers::{Convolvable, Forward, Initialisable};
use crate::tensor::{Shape, Ten};

#[derive(Debug)]
pub struct Pool {
    shape: (usize, usize),
    stride: usize,
    pooling_function: fn(DVector<f32>) -> f32,
}

impl Pool {
    fn new(
        shape: (usize, usize),
        stride: usize,
        pooling_function: fn(DVector<f32>) -> f32,
    ) -> Self {
        Self {
            shape,
            stride,
            pooling_function,
        }
    }
}

impl Initialisable for Pool {
    fn initialise(&mut self, previous_shape: Shape) -> Shape {
        out_shape(previous_shape, self, 1)
    }
}

impl Convolvable for Pool {
    fn shape(&self) -> Shape {
        self.shape.into()
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
        let shape = out_shape(prev_layer.shape, self);
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
