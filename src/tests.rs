#[cfg(test)]
pub mod tests {
    use crate::neural_network::NN;
    use nalgebra::{DMatrix, DVector};
    fn create_nn_for_test () -> (NN, DVector<f32>, DVector<f32>) {
        // Creating network 
        let mut network = match NN::new(3, &[3, 2, 3]) {
            Ok(network) => network,
            Err(e) => { print!("{e:?}"); panic!("noooo") },
        };

        // Setting weights and biases
        network.weights = vec![DMatrix::from_vec(2, 3, vec![-0.75, 0.25, 0.5, -0.5, -0.25, 0.75]), DMatrix::from_vec(3, 2, vec![-0.75, 0.5, -0.25, 0.75, -0.5, 0.25])];
        network.biases = vec![DVector::from_vec(vec![0.0,1.0]), DVector::from_vec(vec![0.75, 0.5, 0.25])];

        // Mock data
        let data: DVector<f32> = DVector::from_vec(vec![0.25, 0.5, 0.75]);
        let expected_output: DVector<f32> = DVector::from_vec(vec![0.0, 1.0, 0.0]);

        (network, data, expected_output)
    }

    #[test]
    fn test_backpropagation() {
        let (mut network, data, expected_output) = self::create_nn_for_test();
        network.layers=NN::forward_pass(&network, &data);
        let (delta_biases, delta_weights) = NN::backpropagation(&network, NN::calculate_cost(&network, &expected_output));

        assert_eq!(delta_biases, vec![DVector::from_vec(vec![-0.18908411,  0.94542056]), DVector::from_vec(vec![0.3588527,  -0.26144633, 0.2913903])]);
        assert_eq!(delta_weights, vec![DMatrix::from_vec(2, 3, vec![-0.04727103, 0.23635514, -0.09454206, 0.47271028, -0.14181308, 0.70906544]), DMatrix::from_vec(3, 2, vec![-0.008971318, 0.0065361583, -0.0072847577, 0.49342248, -0.3594887, 0.40066165])]);
    }

    #[test]
    fn test_cost() {
        let (mut network, data, expected_output) = self::create_nn_for_test();
        network.layers=NN::forward_pass(&network, &data);
        assert_eq!(NN::calculate_cost(&network, &expected_output),  DVector::from_vec(vec![1.0/(1.0+(-1.8_f32).exp()), 1.0/(1.0+(0.2_f32).exp()), 1.0/(1.0+(-0.6_f32).exp())]) - DVector::from_vec(vec![0.0, 1.0, 0.0]))
    }
    
    #[test]
    fn test_forward_pass() {
        let (network, data, _) = self::create_nn_for_test();
        assert_eq!(NN::forward_pass(&network, &data)[1], DVector::from_vec(vec![-0.025, 1.375])); 
        assert_eq!(NN::forward_pass(&network, &data)[2], DVector::from_vec(vec![1.0/(1.0+(-1.8_f32).exp()), 1.0/(1.0+(0.2_f32).exp()), 1.0/(1.0+(-0.6_f32).exp())])); 
    }
    
}
