use nalgebra::{Const, DMatrix, Dyn};

use crate::initialisation::InitialisationOptions;
use crate::layers::convolvable::{im2col, out_shape};
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

struct ConvolutionDelta {
    // For caching, not seperate biases and tensors
    delta_kernels: Vec<(Ten, f32)>,
}

impl std::ops::Add for ConvolutionDelta {
    type Output = ConvolutionDelta;

    fn add(self, rhs: Self) -> Self::Output {
        let mut iterator = std::iter::zip(self.delta_kernels, rhs.delta_kernels);
        let starting_val = match iterator.next() {
            Some(x) => vec![(x.0.0 + x.1.0, x.0.1 + x.1.1)],
            None => {
                return ConvolutionDelta {
                    delta_kernels: Vec::new(),
                };
            }
        };
        ConvolutionDelta {
            delta_kernels: iterator.fold(starting_val, |v, x| {
                [v, vec![(x.0.0 + x.1.0, x.0.1 + x.1.1)]].concat()
            }),
        }
    }
}

impl Backward<ConvolutionDelta> for Convolution {
    fn backprop(
        &self,
        following_layer_derivatives: Ten,
        previous_layer_output: &Ten,
    ) -> (Ten, ConvolutionDelta) {
        todo!()
    }

    fn apply(&mut self, delta: ConvolutionDelta) {
        todo!()
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
        (self.shape.0, self.shape.1, self.kernels.len()).into()
    }
}

impl Convolution {
    fn new(
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
    fn run(&self, prev_layer: Ten) -> Ten {
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
        let biases_mat: DMatrix<f32> = DMatrix::from_iterator(
            self.kernels.len(),
            prev_columnised.ncols(),
            (0..self.kernels.len())
                .map(|x| self.kernels[x].bias)
                .cycle()
                .take(self.kernels.len() * prev_columnised.ncols()),
        );
        let res = ((matrix_of_kernels * &prev_columnised) + biases_mat)
            .transpose()
            .reshape_generic(
                Dyn(self.kernels.len() * prev_columnised.ncols()),
                Const::<1>,
            );
        Ten {
            data: res,
            shape: outshape,
        }
    }
}
