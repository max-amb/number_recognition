use nalgebra::{DMatrix, DVector};
use crate::neural_network::NN;
use crate::training_data::TrainingData;

pub mod neural_network;
pub mod tests;
pub mod training_data;


fn main() {
    let training_data = TrainingData::new("/home/max/Downloads/train-labels.idx1-ubyte", "/home/max/Downloads/train-images.idx3-ubyte");
    let mut network = match NN::new(4, &[784, 16, 16, 10]) {
        Ok(network) => network,
        Err(e) => { print!("{e:?}"); panic!("noooo") },
    };
    
    // let batch_size: i32 = 50;
    // let x = (((training_data.data.len() as f32)/(batch_size as f32)).floor()) as i32;

    let mut avg_cost: f32 = 1.0;
    let mut iter = 0;
    while avg_cost > 0.20 {
        avg_cost = 0.0;
        for i in 0..1000 {
            network.layers = NN::forward_pass(&network, &training_data.data[i]);
            let costs: DVector<f32> = NN::calculate_cost(&network, &training_data.labels[i]);
            let (mut delta_biases_sum, mut delta_weights_sum) = NN::backpropagation(&network, costs);
            for _ in 0..(training_data.data.len()-1) {
                network.layers = NN::forward_pass(&network, &training_data.data[i]);
                let costs: DVector<f32> = NN::calculate_cost(&network, &training_data.labels[i]);
                avg_cost += costs.iter().map(|x| x.powi(2)).sum::<f32>();
                let (delta_biases, delta_weights) = NN::backpropagation(&network, costs);

                for i in &delta_biases {
                    if i.iter().find(|x| x.is_nan() ).is_some() { panic!() };
                };

                for i in &delta_weights {
                    if i.iter().find(|x| x.is_nan() ).is_some() { panic!() };
                };

                delta_biases_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&delta_biases[i]);
                delta_weights_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&delta_weights[i]);
            }
            
            let to_apply_biases: Vec<DVector<f32>> = delta_biases_sum.iter().map(|x| x * 0.000000001).collect();
            let to_apply_weights: Vec<DMatrix<f32>> = delta_weights_sum.iter().map(|x| x * 0.000000001).collect();


            network.weights.iter_mut().enumerate().for_each(|(i,x)| *x += &to_apply_weights[i]);
            network.biases.iter_mut().enumerate().for_each(|(i,x)| *x += &to_apply_biases[i]);
        }
        avg_cost = avg_cost/(training_data.data.len() as f32);
        iter += 1;
        if iter == 1 {
            iter = 0;
            dbg!(avg_cost);
        }
    }
}
