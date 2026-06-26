use nalgebra::{Const, DVector, Dyn};

use crate::layers::primitives::{im2col, out_shape};
use crate::layers::{Convolvable, Forward, Initialisable};
use crate::tensor::{Shape, Ten};

#[derive(Debug)]
pub struct Pool {
    shape: (usize, usize),
    stride: usize,
    pooling_function: fn(DVector<f32>) -> f32,
    output_depth: Option<usize>
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
            output_depth: None
        }
    }
}

impl Initialisable for Pool {
    fn initialise(&mut self, previous_shape: Shape) -> Shape {
        self.output_depth = Some(previous_shape.channels);
        out_shape(previous_shape, self)
    }
}

impl Convolvable for Pool {
    fn out_depth(&self) -> usize {
        self.output_depth.unwrap()
    }

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
    fn run(&self, prev_layer: Ten) -> Ten {
        let outshape = out_shape(prev_layer.shape, self);
        let size_of_view= self.shape.0 * self.shape.1;
        let prev_columnised = im2col(prev_layer, self);
        let mut result: Vec<f32> = Vec::with_capacity(outshape.magnitude());
        /*
        for _ in 0..outshape.channels {
            result.push(Vec::with_capacity(outshape.nrows * outshape.ncols));
        }*/

        for i in 0..outshape.channels {
            let channel_block = prev_columnised.rows(i*size_of_view, size_of_view);
            for col in channel_block.column_iter() {
                result.push((self.pooling_function)(col.into()));
            }
        }
        Ten {
            data: DVector::from_vec(result),
            shape: outshape,
        }
    }
}
