use nalgebra::{Const, DMatrix, Dyn, DVector};

use crate::layers::primitives::{im2col, out_shape};
use crate::layers::{Convolvable, Forward, Mat};

use crate::initialisation::Initialisable;

#[derive(Debug)]
pub struct Kernel {
    pub kernel: DMatrix<f32>,
    pub bias: f32,
    pub stride: usize,
    pub zero_padding: usize,
}

impl Kernel {
    fn new(kernel: DMatrix<f32>, stride: usize, zero_padding: usize) -> Self {
        Self { kernel, bias: 0.0, stride, zero_padding }
    }
}

impl Initialisable for Kernel {
    fn initialise(self, _previous_shape: (usize, usize)) -> Self {
        self       
    } 
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

impl Initialisable for Convolution {
    fn initialise(self, _previous_shape: (usize, usize)) -> Self {
        self 
    } 
}

impl Convolution {
    fn new(kern: Kernel) -> Self {
        Self { filter: kern }
    }

    fn from_kernel(kernel: DMatrix<f32>, stride: usize, zero_padding: usize) -> Self {
        Self { filter: Kernel::new(kernel, stride, zero_padding)  }
    }
}

impl Forward for Convolution {
    fn run(&self, prev_layer: Mat) -> Mat {
        let kern = &self.filter;
        let (new_nrows, new_ncols) = out_shape(prev_layer.shape, kern);

        let prev_columnised = im2col(prev_layer, kern);
        // Cloning kernel isn't horrific, should be somewhat small
        let flattened_kernel = kern
            .kernel
            .clone()
            .reshape_generic(Dyn(1), Dyn(kern.kernel.shape().0 * kern.kernel.shape().1));
        let res = (flattened_kernel * prev_columnised)
            .reshape_generic(Dyn(new_nrows * new_ncols), Const::<1>);
        Mat {
            data: res+DVector::from_element(new_nrows*new_ncols, self.filter.bias),
            shape: (new_nrows, new_ncols),
        }
    }
}
