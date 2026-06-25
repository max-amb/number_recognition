use nalgebra::DVector;

#[derive(Debug)]
pub enum CostFunction {
    Quadratic,
    CategoricalCrossEntropy,
}

impl CostFunction {
    pub fn calculate_cost(&self, observed: &DVector<f32>, expected: &DVector<f32>) -> f32 {
        match self {
            CostFunction::Quadratic => observed
                .iter()
                .enumerate()
                .map(|(i, x)| (x - expected[i]).powi(2))
                .sum::<f32>(),
            CostFunction::CategoricalCrossEntropy => -expected
                .iter()
                .enumerate()
                .map(|(i, x)| x * ((f32::EPSILON + observed[i]).ln()))
                .sum::<f32>(),
        }
    }
}
