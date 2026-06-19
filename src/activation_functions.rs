use nalgebra::DVector;

pub enum ActivationFunction {
    Sigmoid,
    Softmax,
    LeakyRelu { alpha : f32 },
    Relu
}

impl ActivationFunction {
    pub fn apply(&self, layer: DVector<f32>) -> DVector<f32> {
        match self {
            ActivationFunction::Sigmoid => sigmoid(layer),
            ActivationFunction::Relu => relu(layer),
            ActivationFunction::LeakyRelu { alpha }=> leaky_relu(layer, *alpha),
            ActivationFunction::Softmax => softmax(layer)
        }
    }
}

fn sigmoid(layer: DVector<f32>) -> DVector<f32> {
    layer.map(|x| 1.0 / (1.0 + (-x).exp()))
}

fn softmax(layer: DVector<f32>) -> DVector<f32> {
    let layers_max = layer.max();
    let exponentials =
        DVector::from_iterator(layer.nrows(), layer.iter().map(|x| (x - layers_max).exp()));
    &exponentials / exponentials.sum()
}

fn leaky_relu(layer: DVector<f32>, alpha: f32) -> DVector<f32> {
    layer.map(|x| if x.lt(&0.0) { alpha * x } else { x })
}

fn relu(layer: DVector<f32>) -> DVector<f32> {
    layer.map(|x| if x.lt(&0.0) { 0.0 } else { x })
}

pub fn leaky_relu_derivative(input: f32, alpha: f32) -> f32 {
    if input.lt(&0.0) { alpha } else { 1.0 }
}

pub fn relu_derivative(input: f32) -> f32 {
    if input.lt(&0.0) { 0.0 } else { 1.0 }
}
