use nalgebra::{DMatrix, DVector};
use rand::RngExt;
use rand_distr::{Normal, Distribution};

use crate::layers::{Forward, Mat};
use crate::initialisation::{Initialisable, InitialisationOptions};

#[derive(Debug)]
pub struct FullyConnected {
    output_shape: (usize, usize),
    weights: Option<DMatrix<f32>>,
    biases: Option<DVector<f32>>,
    initialisation_options: InitialisationOptions
}

impl FullyConnected {
    fn new(output_shape: (usize, usize), initialisation_options: InitialisationOptions) -> Self {
        Self { output_shape, weights: None, biases: None, initialisation_options } 
    }
}

impl Forward for FullyConnected {
    fn run(&self, prev_layer: Mat) -> Mat {
        let data = self.weights.as_ref().unwrap() * prev_layer.data + self.biases.as_ref().unwrap();
        assert_eq!(data.nrows(), self.output_shape.0 * self.output_shape.1);
        Mat {
            data,
            shape: self.output_shape,
        }
    }
}

impl Initialisable for FullyConnected {
    fn initialise(&mut self, previous_shape: (usize, usize)) -> (usize, usize) {
        let mut rng = rand::rng();

        self.weights = Some(match &self.initialisation_options {
            InitialisationOptions::Random => {
                DMatrix::from_fn(
                    self.output_shape.0 * self.output_shape.1,
                    previous_shape.0 * previous_shape.1,
                    |_, _| rng.random_range(-1.0..=1.0),
                )
            }
            InitialisationOptions::He => {
                let normal_dist = Normal::new(0.0, (2.0_f32 / ((previous_shape.0*previous_shape.1) as f32)).sqrt()).unwrap();
                DMatrix::from_fn(
                    self.output_shape.0 * self.output_shape.1,
                    previous_shape.0 * previous_shape.1,
                    |_, _| {
                        normal_dist.sample(&mut rng)
                    }
                )
            }
        });

        self.biases = Some(DVector::from_element(self.output_shape.0*self.output_shape.1, 0.0));
        self.output_shape
    }
}
