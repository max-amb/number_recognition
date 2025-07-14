use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
pub struct NN {
    pub layers: Vec<DVector<f32>>,
    pub weights: Vec<DMatrix<f32>>,
    pub biases: Vec<DVector<f32>>,
}

impl NN {
    pub fn new (number_of_layers: i32, layer_sizes: &[i32]) -> Result<NN, &'static str> {
        if layer_sizes.len() != number_of_layers as usize {
            return Err("The array of layer sizes has a different number of elements than the number of layers");
        } 

        let mut rng = rand::rng();

        let layers: Vec<DVector<f32>> = (0..number_of_layers)
            .map(|x| DVector::from_element(layer_sizes[x as usize] as usize, 0.0))
            .collect();
        let weights: Vec<DMatrix<f32>> = (1..number_of_layers)
            .map(|x| DMatrix::from_fn(layer_sizes[x as usize] as usize, layer_sizes[(x-1) as usize] as usize, |_, _| {
                let normal_dist = Normal::new(0.0, 2.0/(layer_sizes[(x-1) as usize] as f32)).unwrap();
                normal_dist.sample(&mut rng) }
            )) 
            .collect();
        let biases: Vec<DVector<f32>> = (1..number_of_layers)
            .map(|x| DVector::from_fn(layer_sizes[x as usize] as usize, |_, _| rng.random_range(-1.0..=1.0)))
            .collect();
        return Ok(Self {
            layers,
            weights,
            biases,
        })
    }

    pub fn forward_pass (network: &NN, input: &DVector<f32>) -> Vec<DVector<f32>> {
        let mut new_layers: Vec<DVector<f32>> = vec![ input.clone() ];
        for layer in 0..network.weights.len()-1 {
            new_layers.push((&network.weights[layer]*&new_layers[layer] + &network.biases[layer]).map(|x| NN::leaky_relu(x)));
        }
        new_layers.push((&network.weights[network.weights.len()-1]*&new_layers[network.weights.len()-1] + &network.biases[network.weights.len()-1]).map(|x| NN::sigmoid(x)));
        new_layers
    }

    pub fn backpropagation (network: &NN, costs: DVector<f32>) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
        let mut delta_weights_list: Vec<DMatrix<f32>> = Vec::new(); 
        let mut delta_biases_list: Vec<DVector<f32>> = Vec::new(); 

        let mut current_costs = costs.clone();

        for layer in (0..network.weights.len()).rev() { // Layers
            let mut delta_biases: DVector<f32> = DVector::from_element(current_costs.nrows(), 0.0);

            for row in network.weights[layer].row_iter().enumerate() { // Through the output nodes
                let z_value = network.layers[layer+1][row.0];

                let delta;
                if layer == network.weights.len()-1 {
                    delta= 2.0*current_costs[row.0]*NN::sigmoid_derivative(z_value);
                } else {
                    delta= 2.0*current_costs[row.0]*NN::leaky_relu_derivative(z_value);
                }
                delta_biases[row.0] = delta;
            }
            current_costs = (&network.weights[layer].transpose())*&delta_biases;
            delta_weights_list.push(&delta_biases*(&network.layers[layer].transpose())); 
            delta_biases_list.push(delta_biases);
        }

        (delta_biases_list.into_iter().rev().collect(), delta_weights_list.into_iter().rev().collect())
    }

    pub fn backpropagation_for_parallelisation (network: &NN, costs: DVector<f32>, new_layers: &Vec<DVector<f32>>) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
        let mut delta_weights_list: Vec<DMatrix<f32>> = Vec::new(); 
        let mut delta_biases_list: Vec<DVector<f32>> = Vec::new(); 

        let mut current_costs = costs.clone();

        for layer in (0..network.weights.len()).rev() { // Layers
            let mut delta_biases: DVector<f32> = DVector::from_element(current_costs.nrows(), 0.0);

            for row in network.weights[layer].row_iter().enumerate() { // Through the output nodes
                let z_value = new_layers[layer+1][row.0];

                let delta;
                if layer == network.weights.len()-1 {
                    delta= 2.0*current_costs[row.0]*NN::sigmoid_derivative(z_value);
                } else {
                    delta= 2.0*current_costs[row.0]*NN::leaky_relu_derivative(z_value);
                }
                delta_biases[row.0] = delta;
            }
            current_costs = (&network.weights[layer].transpose())*&delta_biases;
            delta_weights_list.push(&delta_biases*(new_layers[layer].transpose())); 
            delta_biases_list.push(delta_biases);
        }

        (delta_biases_list.into_iter().rev().collect(), delta_weights_list.into_iter().rev().collect())
    }


    pub fn calculate_cost (layers: &Vec<DVector<f32>>, expected_result: &DVector<f32>) -> DVector<f32> {
        &layers[layers.len()-1] - expected_result
    }

    fn sigmoid (input: f32) -> f32 {
        1.0/(1.0+(-input).exp())
    }

    fn sigmoid_derivative (input: f32) -> f32 {
        NN::sigmoid(input)* (1.0-NN::sigmoid(input))
    }

    fn leaky_relu (input: f32) -> f32 {
        if input.lt(&0.0) {
            0.01*input
        } else {
            input
        }
    }

    fn leaky_relu_derivative (input: f32) -> f32 {
        if input.lt(&0.0) {
            0.01
        } else {
            1.0
        }
    }

    pub fn network_classification (layer: &DVector<f32>) -> usize {
        let mut network_classification: (usize, f32) = (usize::MIN, f32::MIN);
        for i in layer.iter().enumerate() {
            if network_classification.1 < *i.1 { network_classification.1 = *i.1; network_classification.0 = i.0 }; 
        };
        network_classification.0
    }
}
