use nalgebra::{Const, DMatrix, Dyn, DVector};

use crate::layers::primitives::{im2col, out_shape};
use crate::layers::{Convolvable, Forward, Mat};

#[derive(Debug)]
pub struct Kernel {
    pub kernel: DMatrix<f32>,
    pub bias: DVector<f32>,
    pub stride: usize,
    pub zero_padding: usize,
}

impl Convolvable for Kernel {
    fn shape(&self) -> (usize, usize) {
        self.kernel.shape()
    }

    fn stride(&self) -> usize {
        self.stride
    }

    fn zero_padding(&self) -> usize {
        self.zero_padding
    }
}

// Begin with a single kernel, extend later
#[derive(Debug)]
pub struct Convolution {
    filter: Kernel,
}

impl Forward for Convolution {
    fn run(&self, prev_layer: Mat) -> Mat {
        let kern = &self.filter;
        let (new_nrows, new_ncols) = out_shape(&prev_layer, kern);

        let prev_columnised = im2col(prev_layer, kern);
        // Cloning kernel isn't horrific, should be somewhat small
        let flattened_kernel = kern
            .kernel
            .clone()
            .reshape_generic(Dyn(1), Dyn(kern.kernel.shape().0 * kern.kernel.shape().1));
        let res = (flattened_kernel * prev_columnised)
            .reshape_generic(Dyn(new_nrows * new_ncols), Const::<1>);
        Mat {
            data: res+&self.filter.bias,
            shape: (new_nrows, new_ncols),
        }
    }
}
