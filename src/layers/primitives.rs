use nalgebra::{DMatrix, DVector, Dyn};
use enum_dispatch::enum_dispatch;

use crate::layers::Activation;
use crate::layers::Convolution;
use crate::layers::FullyConnected;
use crate::layers::Pool;

use crate::tensor::{Shape, Ten};

pub trait Convolvable {
    fn shape(&self) -> (usize, usize);
    fn zero_padding(&self) -> usize;
    fn stride(&self) -> usize;
}

pub fn im2col(m: Ten, conv: &dyn Convolvable) -> DMatrix<f32> {
    let v: DVector<f32> = m.data;
    m.shape.flat_shape();
    let (krows, kcols) = conv.shape();
    assert!(krows <= nrows + conv.zero_padding() && kcols <= ncols + conv.zero_padding());
    let (out_rows, out_cols) = out_shape(in_shape, conv);

    let reshaped: DMatrix<f32> = v.reshape_generic(Dyn(nrows), Dyn(ncols)).resize(
        nrows + conv.zero_padding(),
        ncols + conv.zero_padding(),
        0.0,
    );

    let mut columns = Vec::with_capacity(out_rows * out_cols);

    for i in (0..=((nrows + conv.zero_padding()) - krows)).step_by(conv.stride()) {
        for j in (0..=((ncols + conv.zero_padding()) - kcols)).step_by(conv.stride()) {
            let patch = reshaped.view((i, j), (krows, kcols));
            columns.extend(patch.into_iter().copied());
        }
    }
    DMatrix::from_vec(out_rows, out_cols, columns)
}

pub fn out_shape(in_shape: (usize, usize), conv: &dyn Convolvable) -> (usize, usize) {
    let (nrows, ncols) = in_shape;
    let (crows, ccols) = conv.shape();
    let new_nrows = ((nrows + conv.zero_padding()) - crows) / conv.stride();
    let new_ncols = ((ncols + conv.zero_padding()) - ccols) / conv.stride();
    (new_nrows, new_ncols)
}

#[enum_dispatch]
pub trait Forward {
    fn run(&self, prev_layer: Ten) -> Ten;
}

#[enum_dispatch]
pub trait Initialisable {
    fn initialise(&mut self, previous_shape: Shape) -> Shape;
}

#[derive(Debug)]
#[enum_dispatch(Forward, Initialisable)]
pub enum Layer {
    FC(FullyConnected),
    CONV(Convolution),
    POOL(Pool),
    ACTIVATION(Activation),
}

#[cfg(test)]
mod tests {
    use super::*;
    use hegel::generators as gs;
    use hegel::Generator;
    use hegel::TestCase;

    #[hegel::test]
    fn test_dmatrix_to_matrix(tc: TestCase) {
        let vec: Vec<f32> = tc.draw(gs::vecs(gs::floats()).min_size(1).max_size(200));
        let n_rows = tc.draw(gs::integers().min_value(1).filter(|x| vec.len() % x == 0));
        let n_cols = vec.len() / n_rows;

        assert_eq!(n_rows * n_cols, vec.len());

        let dmat: DMatrix<f32> = DMatrix::from_vec(n_rows, n_cols, vec.clone());
        let mat: Ten = Ten::from(dmat);

        assert_eq!(mat.data, DVector::from_vec(vec));
        assert_eq!(mat.shape, (n_rows, n_cols).into());
    }
}
