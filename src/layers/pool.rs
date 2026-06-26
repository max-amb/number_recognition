use nalgebra::DVector;

use crate::layers::primitives::{im2col, out_shape};
use crate::layers::{Convolvable, Forward, Initialisable};
use crate::tensor::{Shape, Ten};

#[derive(Debug)]
pub enum PoolType {
    MaxPool,
    AvgPool
}

#[derive(Debug)]
pub struct Pool {
    shape: (usize, usize),
    stride: usize,
    pool_type: PoolType,
    output_depth: Option<usize>,
}

impl Pool {
    fn new(
        shape: (usize, usize),
        stride: usize,
        pool_type: PoolType,
    ) -> Self {
        Self {
            shape,
            stride,
            pool_type,
            output_depth: None
        }
    }

    fn pooling_function<I>(&self, values: I) -> f32
        where I: IntoIterator<Item = f32>
    {
        let iterator = values.into_iter();
        match self.pool_type {
            PoolType::MaxPool => iterator.fold(f32::NEG_INFINITY, |curr, x| if x > curr {x} else {curr}),
            PoolType::AvgPool => {
                let (count, sum) = iterator.fold((0, 0.0), |(curr_iter, curr_sum), x| (curr_iter+1, curr_sum+x));
                sum/(count as f32)
            }
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

        for i in 0..outshape.channels {
            let channel_block = prev_columnised.rows(i*size_of_view, size_of_view);
            for col in channel_block.column_iter() {
                result.push(self.pooling_function(col.into_iter().copied()));
            }
        }
        Ten {
            data: DVector::from_vec(result),
            shape: outshape,
        }
    }
}
