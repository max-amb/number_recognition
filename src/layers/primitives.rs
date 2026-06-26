use enum_dispatch::enum_dispatch;
use nalgebra::{DMatrix, DVector, Dyn};

use crate::layers::Activation;
use crate::layers::Convolution;
use crate::layers::FullyConnected;
use crate::layers::Pool;

use crate::tensor::{Shape, Ten};

pub trait Convolvable {
    fn shape(&self) -> Shape;
    fn zero_padding(&self) -> usize;
    fn stride(&self) -> usize;
    fn out_depth(&self) -> usize;
}

pub fn im2col(m: Ten, conv: &dyn Convolvable) -> DMatrix<f32> {
    let outshape = out_shape(m.shape, conv);
    let (nrows, ncols) = m.shape.flat_shape();
    let (krows, kcols) = conv.shape().flat_shape();
    let jump_size = nrows * ncols;

    let mut columns = Vec::with_capacity(outshape.magnitude());
    let reshaped_mats = Vec::from_iter((0..m.shape.magnitude()).step_by(jump_size).map(|x| {
        m.data
            .view((x, 0), (x + jump_size, 1))
            .reshape_generic(Dyn(nrows), Dyn(ncols))
            .resize(
                nrows + conv.zero_padding(),
                ncols + conv.zero_padding(),
                0.0,
            )
    }));

    for j in (0..=((ncols + conv.zero_padding()) - kcols)).step_by(conv.stride()) {
        for i in (0..=((nrows + conv.zero_padding()) - krows)).step_by(conv.stride()) {
            columns.extend(
                (0..conv.out_depth())
                    .map(|x| &reshaped_mats[x])
                    .map(|m| m.view((i, j), (krows, kcols)).into_iter().copied())
                    .flatten(),
            );
        }
    }
    DMatrix::from_vec(
        krows * kcols * conv.out_depth(),
        outshape.nrows * outshape.ncols,
        columns,
    )
}

pub fn out_shape(in_shape: Shape, conv: &dyn Convolvable) -> Shape {
    assert!(
        conv.shape().nrows <= in_shape.nrows + conv.zero_padding()
            && conv.shape().ncols <= in_shape.ncols + conv.zero_padding()
    );
    let new_nrows = ((in_shape.nrows + conv.zero_padding()) - conv.shape().nrows) / conv.stride();
    let new_ncols = ((in_shape.ncols + conv.zero_padding()) - conv.shape().ncols) / conv.stride();
    (new_nrows, new_ncols, conv.out_depth()).into()
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
    use hegel::{Generator, TestCase, HealthCheck};
    use hegel::generators as gs;

    #[hegel::test(suppress_health_check = [HealthCheck::FilterTooMuch])]
    fn test_dmatrix_to_matrix(tc: TestCase) {
        let vec: Vec<f32> = tc.draw(gs::vecs(gs::floats().allow_nan(false)).min_size(1).max_size(200));
        let n_rows = tc.draw(gs::integers().min_value(1).filter(|x| vec.len() % x == 0));
        let n_cols = vec.len() / n_rows;

        assert_eq!(n_rows * n_cols, vec.len());

        let dmat: DMatrix<f32> = DMatrix::from_vec(n_rows, n_cols, vec.clone());
        let ten: Ten = Ten::from(dmat);

        assert_eq!(ten.data, DVector::from_vec(vec));
        assert_eq!(ten.shape, (n_rows, n_cols).into());
    }
}
