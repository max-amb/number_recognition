use nalgebra::{DVector, DMatrix};
use crate::tensor::Shape;

#[derive(Debug, Clone)]
pub struct Ten {
    pub data: DVector<f32>,
    pub shape: Shape
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
}

impl From<DMatrix<f32>> for Ten {
    fn from(mat: DMatrix<f32>) -> Self {
        Self {
            data: DVector::from_vec(mat.into_iter().copied().collect()),
            shape: mat.shape().into(),
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

impl std::ops::Add for &Ten {
    type Output = Ten;
    fn add(self, rhs: &Ten) -> Self::Output {
        assert_eq!(rhs.shape, self.shape);
        let new_data = &self.data + &rhs.data;
        Self::Output {
            data: new_data,
            shape: self.shape
        }
    } 
}

impl std::ops::Mul<Ten> for &Ten {
    type Output = Ten; 
    
    fn mul(self, rhs: Ten) -> Self::Output {
        todo!()
    }
}

impl std::ops::Mul<&Ten> for &Ten {
    type Output = Ten; 
    
    fn mul(self, rhs: &Ten) -> Self::Output {
        todo!()
    }
}

impl std::ops::Mul for Ten {
    type Output = Ten; 

    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
