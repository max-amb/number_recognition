use nalgebra::DVector;
use rand::RngExt;
use rand_distr::{Distribution, Normal};

use crate::tensor::{Shape, Ten};

#[derive(Default, Debug, Copy, Clone)]
pub enum InitialisationOptions {
    Random,
    #[default]
    He,
}

impl InitialisationOptions {
    pub fn create_matrix(&self, shape: Shape) -> Ten {
        let mut rng = rand::rng();

        match self {
            InitialisationOptions::Random => {
                Ten { data: DVector::from_fn(shape.magnitude(),|_, _| rng.random_range(-1.0..=1.0)), shape } 
            }
            InitialisationOptions::He => {
                let normal_dist = Normal::new(0.0, (2.0_f32 / ((shape.magnitude()) as f32)).sqrt()).unwrap();
                Ten { data: DVector::from_fn(
                    shape.magnitude(),
                    |_, _| {
                        normal_dist.sample(&mut rng)
                    }
                ), shape }
            }
        }
    }
}
