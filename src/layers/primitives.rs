use nalgebra::{DMatrix, DVector};

use crate::layers::FullyConnected;
use crate::layers::Convolution;
use crate::layers::Activation;
use crate::layers::Pool;

pub trait Forward {
    fn run(&self, prev_layer: Matrix) -> Matrix;
}

#[derive(Debug)]
pub enum Layer {
    FC(FullyConnected),
    CONV(Convolution),
    POOL(Pool),
    ACTIVATION(Activation)
}

pub struct Matrix {
    pub data: DVector<f32>,
    pub shape: (usize, usize)
}

impl From<DMatrix<f32>> for Matrix {
    fn from(mat: DMatrix<f32>) -> Self {
        Self {
            data: mat.iter().flatten().from_iterator(),
            shape: mat.shape()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hegel::generators as gs;
    use hegel::TestCase;

    #[hegel::test]
    fn test_dmatrix_to_matrix(tc: TestCase) {
        let vec: Vec<f32> = tc.draw(gs::vecs(gs::floats()).min_size(1).max_size(200));
        let n_rows = tc.draw(gs::integers().min_value(1).filter(|x| vec.len() % x == 0));
        let n_cols = vec / n_rows;

        assert_eq!(n_rows*n_cols, vec.len());

        let dmat: DMatrix<f32> = DMatrix::from_vec(n_rows, n_cols, vec);
        let mat: Matrix = Matrix::from(dmat);

        assert_eq!(mat.data, vec);
        assert_eq!(mat.shape, (n_rows, n_cols));
    }
}

