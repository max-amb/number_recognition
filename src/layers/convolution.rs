use nalgebra::{DMatrix, DVector, Dyn};
use std::iter;

use crate::layers::{Forward, Mat};

#[derive(Debug)]
pub struct Kernel {
    kernel: DMatrix<f32>,
    weights: DMatrix<f32>,
    biases: DMatrix<f32>,
    stride: u64,
    zero_padding: u64
}

#[derive(Debug)]
pub struct Convolution {
    filters: Vec<Kernel>
}

impl Forward for Convolution {
    fn run(&self, prev_layer: Mat) -> Mat {
        todo!();
    } 
}

fn im2col(m: Mat, kern: &Kernel) -> DMatrix<f32> {
    let v: DVector<f32> = m.data;
    let (nrows, ncols) = m.shape;
    let (krows, kcols) = kern.kernel.shape();
    let out_rows = krows * kcols;
    let out_cols = (nrows - krows + 1) * (ncols - kcols + 1);

    let reshaped: DMatrix<f32> = v.reshape_generic(Dyn(nrows), Dyn(ncols));

    let mut columns = Vec::with_capacity(out_rows*out_cols);

    for i in 0..=(nrows - krows) {
        for j in 0..=(ncols-kcols) {
            let patch = reshaped.view((i, j), (krows, kcols));
            columns.extend(patch.into_iter().copied());
        }
    }
    DMatrix::from_vec(krows*kcols, (nrows-krows+1)*(ncols-kcols+1), columns)
}
