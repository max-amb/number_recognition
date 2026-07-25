use crate::tensor::Ten;

#[derive(Debug)]
pub enum CostFunction {
    Quadratic,
    CategoricalCrossEntropy,
}

impl CostFunction {
    pub fn calculate_cost(&self, observed: &Ten, expected: &Ten) -> f32 {
        match self {
            CostFunction::Quadratic => observed.data
                .iter()
                .enumerate()
                .map(|(i, x)| (x - expected.data[i]).powi(2))
                .sum::<f32>(),
            CostFunction::CategoricalCrossEntropy => -expected.data
                .iter()
                .enumerate()
                .map(|(i, x)| x * ((f32::EPSILON + observed.data[i]).ln()))
                .sum::<f32>(),
        }
    }
}
