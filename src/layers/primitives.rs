use nalgebra::{DMatrix, DVector, Dyn};
use enum_dispatch::enum_dispatch;

use crate::layers::Activation;
use crate::layers::Convolution;
use crate::layers::FullyConnected;
use crate::layers::Pool;
use crate::initialisation::InitialisationOptions;

pub trait Convolvable {
    fn shape(&self) -> (usize, usize);
    fn zero_padding(&self) -> usize;
    fn stride(&self) -> usize;
}

pub fn im2col(m: Mat, conv: &dyn Convolvable) -> DMatrix<f32> {
    let v: DVector<f32> = m.data;
    let (nrows, ncols) = m.shape;
    let (krows, kcols) = conv.shape();
    assert!(krows <= nrows + conv.zero_padding() && kcols <= ncols + conv.zero_padding());
    let out_rows = krows * kcols;
    let out_cols = (((nrows + conv.zero_padding()) - krows + 1)
        * ((ncols + conv.zero_padding()) - kcols + 1))
        / conv.stride();

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
    fn run(&self, prev_layer: Mat) -> Mat;
}

#[derive(Debug)]
#[enum_dispatch(Forward)]
pub enum Layer {
    FC(FullyConnected),
    CONV(Convolution),
    POOL(Pool),
    ACTIVATION(Activation),
}

pub struct Mat {
    pub data: DVector<f32>,
    pub shape: (usize, usize),
}

impl Mat {
    pub fn fmap(self, function: impl Fn(DVector<f32>) -> DVector<f32>) -> Self {
        let new_data = function(self.data);
        assert_eq!(new_data.shape(), self.shape);
        Self {
            data: new_data,
            shape: self.shape,
        }
    }
}

impl From<DMatrix<f32>> for Mat {
    fn from(mat: DMatrix<f32>) -> Self {
        Self {
            data: DVector::from_vec(mat.into_iter().copied().collect()),
            shape: mat.shape(),
        }
    }
}
//
// impl Deref for Mat {
//     type Target = DVector<f32>;
//     fn deref(&self) -> &Self::Target {
//         &self.data
//     }
// }
//
// impl DerefMut for Mat {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.data
//     }
// }
//
// impl IntoIterator for Mat  {
//     type Item = f32;
//     type IntoIter = std::vec::IntoIter<Self::Item>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         let as_vec: Vec<f32> = self.data.into_iter().copied().collect();
//         as_vec.into_iter()
//     }
// }

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
        let mat: Mat = Mat::from(dmat);

        assert_eq!(mat.data, DVector::from_vec(vec));
        assert_eq!(mat.shape, (n_rows, n_cols));
    }
}
