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
    pub fn fmap(&self, function: impl Fn(&DVector<f32>) -> DVector<f32>) -> Self {
        let new_data: DVector<f32> = function(&self.data);
        assert_eq!(new_data.len(), self.shape.magnitude());
        Self {
            data: new_data,
            shape: self.shape,
        }
    }

    pub fn transpose(&self) -> Self {
        let matrix_magnitude = self.shape.nrows * self.shape.ncols;
        let output_shape: Shape = (self.shape.ncols, self.shape.nrows, self.shape.channels).into();
        let mut result: Vec<f32> = Vec::with_capacity(output_shape.magnitude());
        // This is viable because reshape_generic is zero cost, https://docs.rs/nalgebra/latest/nalgebra/base/struct.Matrix.html#method.reshape_generic.
        // And so is collection.
        for channel in (0..self.shape.channels).map(|x| x * matrix_magnitude) {
            let mat_as_vec = self.data.view((channel, 0), (matrix_magnitude, 1));
            let mat_transposed = mat_as_vec
                .reshape_generic(Dyn(self.shape.nrows), Dyn(self.shape.ncols))
                .transpose();
            result.extend(mat_transposed.into_iter().copied());
        }
        Self {
            data: DVector::from_vec(result),
            shape: output_shape,
        }
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

impl From<&DVector<f32>> for Ten {
    fn from(mat: &DVector<f32>) -> Self {
        Ten {
            data: DVector::from_vec(mat.into_iter().copied().collect()),
            shape: mat.shape().into(),
        }
    }
}

impl From<Vec<DMatrix<f32>>> for Ten {
    fn from(matrices: Vec<DMatrix<f32>>) -> Self {
        assert!(!matrices.is_empty());
        assert!(
            matrices
                .iter()
                .map(|x| x.shape())
                .all(|x| x == matrices[0].shape())
        );
        let output_shape: Shape = (matrices[0].nrows(), matrices[0].ncols(), matrices.len()).into();
        Self {
            data: DVector::from_iterator(
                output_shape.magnitude(),
                matrices
                    .into_iter()
                    .flat_map(|x| x.into_iter().copied().collect::<Vec<f32>>()),
            ),
            shape: output_shape,
        }
    }
}

impl From<DVector<f32>> for Ten {
    fn from(v: DVector<f32>) -> Self {
        let shape: Shape = v.shape().into();
        Self { data: v, shape }
    }
}

impl std::ops::Mul<f32> for &Ten {
    type Output = Ten;
    fn mul(self, rhs: f32) -> Self::Output {
        let new_data = &self.data * rhs;
        Self::Output {
            data: new_data,
            shape: self.shape,
        }
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

impl std::ops::Sub for &Ten {
    type Output = Ten;
    fn sub(self, rhs: &Ten) -> Self::Output {
        assert_eq!(rhs.shape, self.shape);
        let new_data = &self.data - &rhs.data;
        Self::Output {
            data: new_data,
            shape: self.shape,
        }
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

impl std::ops::Sub for Ten {
    type Output = Ten;
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl std::ops::Sub<Ten> for &Ten {
    type Output = Ten;
    fn sub(self, rhs: Ten) -> Self::Output {
        self - &rhs
    }
}

impl std::ops::Sub<&Ten> for Ten {
    type Output = Ten;
    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}

impl std::ops::Mul<f32> for Ten {
    type Output = Ten;
    fn mul(self, rhs: f32) -> Self::Output {
        &self * rhs
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

#[cfg(test)]
mod tests {
    use super::*;
    use hegel::generators as gs;
    use hegel::{Generator, HealthCheck, TestCase};

    // Need to supress filter too mutch as we have a very restrictive filter
    #[hegel::test(suppress_health_check = [HealthCheck::FilterTooMuch])]
    fn test_dmatrix_to_tensor(tc: TestCase) {
        let vec: Vec<f32> = tc.draw(
            gs::vecs(gs::floats().allow_nan(false))
                .min_size(1)
                .max_size(200),
        );
        let n_rows = tc.draw(
            gs::integers()
                .min_value(1)
                .max_value(vec.len())
                .filter(|x| vec.len().is_multiple_of(*x)),
        );
        let n_cols = vec.len() / n_rows;

        assert_eq!(n_rows * n_cols, vec.len());

        let dmat: DMatrix<f32> = DMatrix::from_vec(n_rows, n_cols, vec.clone());
        let ten: Ten = Ten::from(dmat);

        assert_eq!(ten.data, DVector::from_vec(vec));
        assert_eq!(ten.shape, (n_rows, n_cols).into());
    }

    #[hegel::test(test_cases = 10000)]
    fn test_vec_of_dmatrix_to_tensor(tc: TestCase) {
        let vec: Vec<f32> = tc.draw(
            gs::vecs(gs::floats().allow_nan(false))
                .min_size(1)
                .max_size(2000),
        );
        let nrows = tc.draw(
            gs::integers()
                .min_value(1)
                .max_value(vec.len())
                .filter(|x| vec.len().is_multiple_of(*x)),
        );
        let ncols = tc.draw(
            gs::integers()
                .min_value(1)
                .max_value(vec.len())
                .filter(|x| (vec.len() / nrows).is_multiple_of(*x)),
        );
        let channels = vec.len() / (nrows * ncols);

        assert_eq!(nrows * ncols * channels, vec.len());

        let mut to_convert: Vec<DMatrix<f32>> = Vec::new();
        for channel in 0..channels {
            to_convert.push(DMatrix::from_iterator(
                nrows,
                ncols,
                (0..nrows * ncols).map(|x| vec[x + (channel * nrows * ncols)]),
            ));
        }
        let ten: Ten = Ten::from(to_convert);

        assert_eq!(ten.data, DVector::from_vec(vec));
        assert_eq!(ten.shape, (nrows, ncols, channels).into());
    }
}
