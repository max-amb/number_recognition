use nalgebra::{Const, DMatrix, DVector, Dyn};

use crate::layers::{Forward, Mat};

#[derive(Debug)]
pub struct Kernel {
    pub kernel: DMatrix<f32>,
    pub biases: DMatrix<f32>,
    pub stride: usize,
    pub zero_padding: usize
}

// Begin with a single kernel, extend later
#[derive(Debug)]
pub struct Convolution {
    filter: Kernel
}

impl Forward for Convolution {
    fn run(&self, prev_layer: Mat) -> Mat {
        let kern = &self.filter;
        let (nrows, ncols) = prev_layer.shape();
        let (krows, kcols) = kern.kernel.shape();
        let new_nrows = ((nrows+kern.zero_padding) - krows)/kern.stride;
        let new_ncols = ((ncols+kern.zero_padding) - kcols)/kern.stride;

        let prev_columnised = im2col(prev_layer, kern);
        // Cloning kernel isn't horrific, should be somewhat small
        let flattened_kernel = kern.kernel.clone().reshape_generic(Dyn(1), Dyn(kern.kernel.shape().0 * kern.kernel.shape().1));
        let res = (flattened_kernel*prev_columnised).reshape_generic(Dyn(new_nrows*new_ncols), Const::<1>);
        Mat { data: res, shape: (new_nrows, new_ncols) }
    } 
}

fn im2col(m: Mat, kern: &Kernel) -> DMatrix<f32> {
    let v: DVector<f32> = m.data;
    let (nrows, ncols) = m.shape;
    let (krows, kcols) = kern.kernel.shape();
    assert!(krows <= nrows+kern.zero_padding && kcols <= ncols+kern.zero_padding);
    let out_rows = krows * kcols;
    let out_cols = (((nrows+kern.zero_padding) - krows + 1) * ((ncols+kern.zero_padding) - kcols + 1))/kern.stride;

    let reshaped: DMatrix<f32> = v.reshape_generic(Dyn(nrows), Dyn(ncols)).resize(nrows+kern.zero_padding, ncols+kern.zero_padding, 0.0);

    let mut columns = Vec::with_capacity(out_rows*out_cols);

    for i in (0..=((nrows+kern.zero_padding) - krows)).step_by(kern.stride) {
        for j in (0..=((ncols+kern.zero_padding)-kcols)).step_by(kern.stride) {
            let patch = reshaped.view((i, j), (krows, kcols));
            columns.extend(patch.into_iter().copied());
        }
    }
    DMatrix::from_vec(out_rows, out_cols, columns)
}
