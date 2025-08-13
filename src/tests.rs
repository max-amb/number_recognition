#[cfg(test)]
pub mod test {
    use crate::neural_network::{CostFunction, InitialisationOptions, NN};
    use nalgebra::{DMatrix, DVector};
    use float_cmp::assert_approx_eq;
    fn create_nn_for_test () -> (NN, DVector<f32>, DVector<f32>) {
        // Creating network 
        let mut network = match NN::new(3, &[3, 2, 3], InitialisationOptions::Random, Some(0.2)) {
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
        let (network, data, expected_output) = self::create_nn_for_test();
        let new_layers = NN::forward_pass(&network, &data, &CostFunction::Quadratic);
        let (delta_biases, delta_weights) = NN::backprop(&network, &expected_output, &new_layers, &CostFunction::Quadratic);

        let expected_output = [DVector::from_vec(vec![-0.0733288, 0.366644]), DVector::from_vec(vec![0.208924,-0.272186, 0.295432])];
        for i in delta_biases.iter().enumerate() {
            for j in i.1.iter().enumerate() {
                assert_approx_eq!(f32, *j.1, expected_output[i.0][j.0] , epsilon = 0.000001);
            }
        }

        let expected_output = [DMatrix::from_vec(2, 3, vec![-0.0183322, 0.091661, -0.0366644, 0.183322, -0.0549966, 0.274983]), DMatrix::from_vec(3, 2, vec![-0.0052231, 0.00680465, -0.0073858, 0.287271, -0.374256, 0.406219])];
        dbg!(&delta_weights);
        for i in delta_weights.iter().enumerate() {
            for j in i.1.row_iter().enumerate() {
                for k in j.1.iter().enumerate() {
                    assert_approx_eq!(f32, *k.1, expected_output[i.0][(j.0, k.0)], epsilon = 0.000001);
                }
            }
        }
    }

    #[test]
    fn test_cost() {
        let (mut network, data, expected_output) = self::create_nn_for_test();
        network.layers=NN::forward_pass(&network, &data, &CostFunction::Quadratic);
        assert_eq!(NN::calculate_cost(&network.layers, &expected_output),  DVector::from_vec(vec![1.0/(1.0+(-1.8_f32).exp()), 1.0/(1.0+(0.2_f32).exp()), 1.0/(1.0+(-0.6_f32).exp())]) - DVector::from_vec(vec![0.0, 1.0, 0.0]))
    }
    
    #[test]
    fn test_forward_pass() {
        let (network, data, _) = self::create_nn_for_test();
        assert_eq!(NN::forward_pass(&network, &data, &CostFunction::Quadratic)[1], DVector::from_vec(vec![-0.025, 1.375])); 
        assert_eq!(NN::forward_pass(&network, &data, &CostFunction::Quadratic)[2], DVector::from_vec(vec![1.0/(1.0+(-1.8_f32).exp()), 1.0/(1.0+(0.2_f32).exp()), 1.0/(1.0+(-0.6_f32).exp())])); 
    }
    
}
