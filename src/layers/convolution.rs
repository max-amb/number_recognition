use nalgebra::{Const, DMatrix, DVector, Dyn};

use crate::initialisation::InitialisationOptions;
use crate::layers::convolvable::{col2im, im2col, out_shape};
use crate::layers::primitives::Delta;
use crate::layers::{Backward, Convolvable, Forward, Initialisable};
use crate::tensor::{Shape, Ten};

#[derive(Debug)]
pub struct Kernel {
    pub kernel: Option<Ten>,
    pub bias: f32,
    shape: (usize, usize),
}

impl Kernel {
    fn new(shape: (usize, usize)) -> Self {
        Self {
            kernel: None,
            bias: 0.0,
            shape,
        }
    }

    fn initialise(&mut self, previous_shape: Shape, initialisation_options: InitialisationOptions) {
        self.kernel = Some(initialisation_options.create_matrix(
            (self.shape.0, self.shape.1, previous_shape.channels).into(),
            previous_shape.magnitude(),
        ));
    }
}

#[derive(Debug)]
pub struct Convolution {
    kernels: Vec<Kernel>,
    stride: usize,
    zero_padding: usize,
    initialisation_options: InitialisationOptions,
    shape: (usize, usize),
}

// For caching, not seperate biases and tensors
#[derive(Debug)]
pub struct ConvolutionDelta(pub Vec<(Ten, f32)>);

impl std::ops::Add for ConvolutionDelta {
    type Output = ConvolutionDelta;

    fn add(self, rhs: Self) -> Self::Output {
        let iterator = std::iter::zip(self.0, rhs.0);
        ConvolutionDelta(iterator.map(|(l, r)| (l.0 + r.0, l.1 + r.1)).collect())
    }
}

impl std::ops::Mul<f32> for ConvolutionDelta {
    type Output = ConvolutionDelta; 

    fn mul(self, rhs: f32) -> Self::Output {
        ConvolutionDelta(self.0.into_iter().map(|x| (x.0 * rhs, x.1*rhs) ).collect())
    }
}

impl Backward for Convolution {
    fn final_layer_backprop(&self,cost_func_derivative: Ten, previous_layer_output: &Ten) -> (Ten,Delta) {
        panic!();
    }

    fn backprop(
        &self,
        following_layer_derivatives: Ten,
        previous_layer_output: &Ten,
    ) -> (Ten, Delta) {
        let outshape = out_shape(previous_layer_output.shape, self);
        let flattened_derivatives: DMatrix<f32> = DMatrix::from_row_iterator(
            outshape.channels,
            outshape.nrows * outshape.ncols,
            following_layer_derivatives.data.into_iter().copied(),
        );
        let prev_columnised = im2col(previous_layer_output, self).transpose();
        let filter_derivatives = &flattened_derivatives * prev_columnised;

        // Filter derivatives must be of shape (num of kernels) x (size of kernel)
        assert_eq!(
            filter_derivatives.shape(),
            (
                outshape.channels,
                self.shape.0 * self.shape.1 * previous_layer_output.shape.channels,
            )
        );

        let bias: Vec<f32> = {
            use itertools::Itertools;
            let iterator_jump = outshape.flat_shape().0 * outshape.flat_shape().1;
            Vec::from_iter(
                following_layer_derivatives
                    .data
                    .into_iter()
                    .copied()
                    .chunks(iterator_jump)
                    .into_iter()
                    .map(|c| c.sum::<f32>()),
            )
        };

        let delta = Vec::from_iter(std::iter::zip(filter_derivatives.row_iter(), bias).map(
            |(f, b)| {
                let shape: Shape = (self.shape.0, self.shape.1, previous_layer_output.shape.channels).into();
                (
                    Ten {
                        data: DVector::from_iterator(shape.magnitude(), f.into_iter().copied()),
                        shape,
                    },
                    b,
                )
            },
        ));

        let flattened_kernels = DMatrix::from_iterator(
            self.shape.0 * self.shape.1 * previous_layer_output.shape.channels,
            self.kernels.len(),
            self.kernels
                .iter()
                .flat_map(|x| x.kernel.as_ref().unwrap().data.into_iter().copied()),
        );
        let current_layer_derivatives: Ten = col2im(
            &(flattened_kernels * flattened_derivatives),
            self,
            previous_layer_output.shape,
        );

        (current_layer_derivatives, Delta::CONVD(ConvolutionDelta(delta)))
    }

    fn apply(&mut self, delta: Delta) {
        if let Delta::CONVD(delta) = delta {
            self.kernels
                .iter_mut()
                .enumerate()
                .for_each(|(i, Kernel { kernel, bias, .. })| {
                    *kernel = Some(kernel.as_ref().unwrap() - &delta.0[i].0);
                    *bias -= delta.0[i].1;
                });
        } else {
            panic!();
        }
    }
}

impl Convolvable for Convolution {
    fn filter_shape(&self) -> Shape {
        self.shape.into()
    }

    fn stride(&self) -> usize {
        self.stride
    }

    fn zero_padding(&self) -> usize {
        self.zero_padding
    }

    fn out_depth(&self) -> usize {
        self.kernels.len()
    }
}

impl Initialisable for Convolution {
    fn initialise(&mut self, previous_shape: Shape) -> Shape {
        for kern in &mut self.kernels {
            kern.initialise(previous_shape, self.initialisation_options);
        }
        out_shape(previous_shape, self)
    }
}

impl Convolution {
    pub fn new(
        num_of_kernels: usize,
        shape: (usize, usize),
        stride: usize,
        zero_padding: usize,
        initialisation_options: InitialisationOptions,
    ) -> Self {
        Self {
            kernels: Vec::from_iter((0..num_of_kernels).map(|_| Kernel::new(shape))),
            stride,
            zero_padding,
            initialisation_options,
            shape,
        }
    }
}

impl Forward for Convolution {
    fn run(&self, prev_layer: &Ten) -> Ten {
        let outshape = out_shape(prev_layer.shape, self);
        let previous_layers_channels = prev_layer.shape.channels;

        let prev_columnised = im2col(&prev_layer, self);
        let mut rows_of_kernels: Vec<f32> = Vec::with_capacity(
            self.shape.0 * self.shape.1 * self.kernels.len() * previous_layers_channels,
        );

        for kern in &self.kernels {
            rows_of_kernels.extend(kern.kernel.as_ref().unwrap().data.into_iter().copied());
        }

        let matrix_of_kernels = DMatrix::from_row_iterator(
            self.kernels.len(),
            self.shape.0 * self.shape.1 * previous_layers_channels,
            rows_of_kernels,
        );

        let mut res = matrix_of_kernels * &prev_columnised; // (num_kernels x num_positions)

        let biases: DVector<f32> = DVector::from_iterator(
            self.kernels.len(),
            self.kernels.iter().map(|k| k.bias),
        );

        res.column_iter_mut().for_each(|mut col| col += &biases);

        Ten {
            data: 
        res
            .transpose()
            .reshape_generic(
                Dyn(self.kernels.len() * prev_columnised.ncols()),
                Const::<1>,
            ),
            shape: outshape,
        }
    }
}
