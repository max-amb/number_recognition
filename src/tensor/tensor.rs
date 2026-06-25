use crate::tensor::Shape;
use nalgebra::{DMatrix, DVector, Dyn};

/// Data is stored in column major order, channel by channel, i.e. the entire matrix for a single
/// channel is stored together
#[derive(Debug, Clone)]
pub struct Ten {
    pub data: DVector<f32>,
    pub shape: Shape,
}

impl Ten {
    pub fn fmap(self, function: impl Fn(DVector<f32>) -> DVector<f32>) -> Self {
        let new_data: DVector<f32> = function(self.data);
        assert_eq!(new_data.len(), self.shape.magnitude());
        Self {
            data: new_data,
            shape: self.shape,
        }
    }

    pub fn new() -> Self {
        todo!()
    }
}

impl From<DMatrix<f32>> for Ten {
    fn from(mat: DMatrix<f32>) -> Self {
        Self {
            data: DVector::from_vec(mat.into_iter().copied().collect()),
            shape: mat.shape().into(),
        }
    }
}

impl From<DVector<f32>> for Ten {
    fn from(v: DVector<f32>) -> Self {
        let shape: Shape = v.shape().into();
        Self { data: v, shape }
    }
}

impl std::ops::Add for Ten {
    type Output = Ten;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl std::ops::Add<Ten> for &Ten {
    type Output = Ten;
    fn add(self, rhs: Ten) -> Self::Output {
        self + &rhs
    }
}

impl std::ops::Add<&Ten> for Ten {
    type Output = Ten;
    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

impl std::ops::Add for &Ten {
    type Output = Ten;
    fn add(self, rhs: &Ten) -> Self::Output {
        assert_eq!(rhs.shape, self.shape);
        let new_data = &self.data + &rhs.data;
        Self::Output {
            data: new_data,
            shape: self.shape,
        }
    }
}

impl std::ops::Mul<Ten> for &Ten {
    type Output = Ten;

    fn mul(self, rhs: Ten) -> Self::Output {
        self * &rhs
    }
}

impl std::ops::Mul<&Ten> for Ten {
    type Output = Ten;

    fn mul(self, rhs: &Ten) -> Self::Output {
        &self * rhs
    }
}

/// Channel by channel multiplication of matrices
impl std::ops::Mul for &Ten {
    type Output = Ten;

    fn mul(self, rhs: &Ten) -> Self::Output {
        assert_eq!(self.shape.channels, rhs.shape.channels);
        assert_eq!(self.shape.ncols, rhs.shape.nrows);
        let new_shape: Shape = (self.shape.nrows, rhs.shape.ncols, self.shape.channels).into();
        let self_size = self.shape.nrows * self.shape.ncols;
        let rhs_size = rhs.shape.nrows * rhs.shape.ncols;

        let mut res: Vec<f32> = Vec::with_capacity(new_shape.magnitude());
        for i in 0..self.shape.channels {
            let self_view = self
                .data
                .view((i * (self_size), 0), (self_size, 1))
                .reshape_generic(Dyn(self.shape.nrows), Dyn(self.shape.ncols));
            let rhs_view = rhs
                .data
                .view((i * rhs_size, 0), (rhs_size, 1))
                .reshape_generic(Dyn(rhs.shape.nrows), Dyn(rhs.shape.ncols));
            res.extend((self_view * rhs_view).into_iter().copied());
        }

        Self::Output {
            data: DVector::from_vec(res),
            shape: new_shape,
        }
    }
}

impl std::ops::Mul for Ten {
    type Output = Ten;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}
