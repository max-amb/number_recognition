use nalgebra::DVector;

pub fn sigmoid (input: f32) -> f32 {
    1.0/(1.0+(-input).exp())
}

pub fn softmax(layer: DVector<f32>) -> DVector<f32> {
    let layers_max = layer.max();
    let exponentials = DVector::from_iterator(layer.nrows(), layer.iter().map(|x| (x-layers_max).exp()));
    &exponentials/exponentials.sum()
}

pub fn leaky_relu (input: f32, alpha: f32) -> f32 {
    if input.lt(&0.0) {
        alpha*input
    } else {
        input
    }
}

pub fn leaky_relu_derivative (input: f32, alpha: f32) -> f32 {
    if input.lt(&0.0) {
        alpha
    } else {
        1.0
    }
}

pub fn relu (input: f32) -> f32 {
    if input.lt(&0.0) {
        0.0
    } else {
        input
    }
}

pub fn relu_derivative (input: f32) -> f32 {
    if input.lt(&0.0) {
        0.0
    } else {
        1.0
    }
}
