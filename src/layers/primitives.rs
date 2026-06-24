use nalgebra::{DMatrix, DVector, Dyn};
use std::ops::{Deref, DerefMut};

use crate::layers::FullyConnected;
use crate::layers::{Convolution, Kernel};
use crate::layers::Activation;
use crate::layers::Pool;

pub trait Forward {
    fn run(&self, prev_layer: Mat) -> Mat;
}

#[derive(Debug)]
pub enum Layer {
    FC(FullyConnected),
    CONV(Convolution),
    POOL(Pool),
    ACTIVATION(Activation)
}

pub struct Mat {
    pub data: DVector<f32>,
    pub shape: (usize, usize)
}

impl Mat {
    pub fn fmap(self, function: impl Fn(DVector<f32>) -> DVector<f32>) -> Self {
        let new_data = function(self.data);
        assert_eq!(new_data.shape(), self.shape);
        Self {
            data: new_data,
            shape: self.shape
        }
    }
}

impl From<DMatrix<f32>> for Mat {
    fn from(mat: DMatrix<f32>) -> Self {
        Self {
            data: DVector::from_vec(mat.into_iter().copied().collect()),
            shape: mat.shape()
        }
    }
}

impl Deref for Mat {
    type Target = DVector<f32>;  
    fn deref(&self) -> &Self::Target {
        &self.data 
    }
}

impl DerefMut for Mat {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data 
    }
}

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

        assert_eq!(n_rows*n_cols, vec.len());

        let dmat: DMatrix<f32> = DMatrix::from_vec(n_rows, n_cols, vec.clone());
        let mat: Mat = Mat::from(dmat);

        assert_eq!(mat.data, DVector::from_vec(vec));
        assert_eq!(mat.shape, (n_rows, n_cols));
    }
}

