use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::sync::{mpsc, Arc};
use std::thread;

use crate::training_data::TrainingData;

#[derive(PartialEq)]
pub enum InitialisationOptions {
    Random,
    He,
}

impl Default for InitialisationOptions {
    fn default() -> Self {
        InitialisationOptions::He
    }
}

#[derive(Debug, Clone)]
pub struct NN {
    pub layers: Vec<DVector<f32>>,
    pub weights: Vec<DMatrix<f32>>,
    pub biases: Vec<DVector<f32>>,
    pub alpha: f32,
}

impl NN {
    pub fn new (number_of_layers: i32, layer_sizes: &[i32], initialisation: InitialisationOptions, alpha_value: Option<f32>) -> Result<NN, &'static str> {
        if layer_sizes.len() != number_of_layers as usize {
            return Err("The array of layer sizes has a different number of elements than the number of layers");
        } 

        let mut rng = rand::rng();

        let layers: Vec<DVector<f32>> = (0..number_of_layers)
            .map(|x| DVector::from_element(layer_sizes[x as usize] as usize, 0.0))
            .collect();

        let weights: Vec<DMatrix<f32>> = match initialisation {
            InitialisationOptions::Random => {
                (1..number_of_layers)
                    .map(|x| DMatrix::from_fn(layer_sizes[x as usize] as usize, layer_sizes[(x-1) as usize] as usize, |_, _| rng.random_range(-1.0..=1.0)))
                    .collect()},
            InitialisationOptions::He => {
                (1..number_of_layers)
                .map(|x| DMatrix::from_fn(layer_sizes[x as usize] as usize, layer_sizes[(x-1) as usize] as usize, |_, _| {
                    let normal_dist = Normal::new(0.0, 2.0/(layer_sizes[(x-1) as usize] as f32)).unwrap();
                    normal_dist.sample(&mut rng) }
                )) 
                .collect() },
        };

        let biases: Vec<DVector<f32>> = (1..number_of_layers)
            .map(|x| DVector::from_fn(layer_sizes[x as usize] as usize, |_, _| rng.random_range(-1.0..=1.0)))
            .collect();

        let alpha = match alpha_value { Some(val) => val, _ => 0.01};
        return Ok(Self {
            layers,
            weights,
            biases,
            alpha,
        })
    }

    pub fn forward_pass (network: &NN, input: &DVector<f32>) -> Vec<DVector<f32>> {
        let mut new_layers: Vec<DVector<f32>> = vec![ input.clone() ];
        for layer in 0..network.weights.len()-1 {
            new_layers.push((&network.weights[layer]*&new_layers[layer] + &network.biases[layer]).map(|x| NN::leaky_relu(x, network.alpha)));
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
                    delta= 2.0*current_costs[row.0]*NN::leaky_relu_derivative(z_value, network.alpha);
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
                    delta= 2.0*current_costs[row.0]*NN::leaky_relu_derivative(z_value, network.alpha);
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

    fn leaky_relu (input: f32, alpha: f32) -> f32 {
        if input.lt(&0.0) {
            alpha*input
        } else {
            input
        }
    }

    fn leaky_relu_derivative (input: f32, alpha: f32) -> f32 {
        if input.lt(&0.0) {
            alpha
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

    pub fn output_model_to_file (network: &NN, path: &str) -> std::io::Result<()> {
        let mut f = OpenOptions::new()
            .write(true)
            .read(false)
            .create_new(true)
            .open(path)?;

        let mut output: String = String::from("[layers]\n");
        output.push_str(&network.layers.iter().fold(String::from("["), |s, x| format!("{s}{}, ", x.len())));
        output.replace_range(output.len()-2..output.len(), "]\n");

        output.push_str("\n[weights]");
        for weight in network.weights.iter().enumerate() {
            output.push_str(&format!("\n["));
            for individual_number in weight.1 {
                output.push_str(&format!("{}, ", individual_number));
            }
            output.replace_range(output.len()-2..output.len(), "]");
        }

        output.push_str("\n\n[biases]");
        for biase in network.biases.iter().enumerate() {
            output.push_str(&format!("\n["));
            for individual_number in biase.1 {
                output.push_str(&format!("{}, ", individual_number));
            }
            output.replace_range(output.len()-2..output.len(), "]");
        }

        output.push_str("\n\n[alpha]");
        output.push_str(&format!("{}\n", network.alpha));

        write!(f, "{}", output)
    }

    pub fn generate_model_from_file (path: &str) -> Result<NN, std::io::Error> {
        let f = OpenOptions::new()
            .read(true)
            .open(path)?;
        let mut reader = BufReader::new(f);
        let mut line = String::new();

        // Layers
        reader.read_line(&mut line)?;
        assert_eq!(&line, "[layers]\n");
        line.clear();
        reader.read_line(&mut line)?;
        line = line.strip_prefix("[").unwrap().to_string();
        line = line.strip_suffix("]\n").unwrap().to_string();
        let layer_sizes: Vec<usize> = line.split(", ").map(|x| x.parse::<usize>().unwrap()).collect();
        line.clear();

        let layers: Vec<DVector<f32>> = (0..layer_sizes.len())
            .map(|x| DVector::from_element(layer_sizes[x as usize] as usize, 0.0))
            .collect();

        // Weights
        let mut weights: Vec<DMatrix<f32>> = Vec::new();
        reader.read_line(&mut line)?;
        assert_eq!(&line, "\n");
        line.clear();
        reader.read_line(&mut line)?;
        assert_eq!(&line, "[weights]\n");
        line.clear();

        for i in 0..layer_sizes.len()-1 {
            reader.read_line(&mut line)?;
            line = line.strip_prefix("[").unwrap().to_string();
            line = line.strip_suffix("]\n").unwrap().to_string();
            weights.push(DMatrix::from_iterator(layer_sizes[i+1], layer_sizes[i], line.split(", ").map(|x| x.parse::<f32>().unwrap())));
            line.clear();
        }

        // Biases
        let mut biases: Vec<DVector<f32>> = Vec::new();
        reader.read_line(&mut line)?;
        assert_eq!(&line, "\n");
        line.clear();
        reader.read_line(&mut line)?;
        assert_eq!(&line, "[biases]\n");
        line.clear();

        for i in 0..layer_sizes.len()-1 {
            reader.read_line(&mut line)?;
            line = line.strip_prefix("[").unwrap().to_string();
            line = line.strip_suffix("]\n").unwrap().to_string();
            biases.push(DVector::from_iterator(layer_sizes[i+1], line.split(", ").map(|x| x.parse::<f32>().unwrap())));
            line.clear();
        }

        // Alpha
        reader.read_line(&mut line)?;
        assert_eq!(&line, "\n");
        line.clear();
        reader.read_line(&mut line)?;
        assert_eq!(&line, "[alpha]\n");
        line.clear();
        reader.read_line(&mut line)?;
        let alpha: f32 = line.parse().unwrap();


        Ok(NN {
            layers,
            weights,
            biases,
            alpha
        })
    }

    pub fn training (mut network: NN, cycle_size: usize, training_data: TrainingData, precision: f32) -> NN {
        assert!(cycle_size<training_data.data.len());

        let training_data_ref = Arc::new(training_data);
        let mut avg_score: f32 = 0.0;
        let mut iterator_over_cycles = cycle_size;

        while avg_score < precision {
            let network_for_this_iter = Arc::new(network.clone());
            let (tx, rx) = mpsc::channel();
            let mut handles = vec![];
            avg_score = 0.0;

            for i in iterator_over_cycles-cycle_size..iterator_over_cycles {
                let tx_cloned = tx.clone();
                let new_network = Arc::clone(&network_for_this_iter);
                let training_data_cloned = Arc::clone(&training_data_ref);

                let handle = thread::spawn(move || {
                    let new_layers = NN::forward_pass(&new_network, &training_data_cloned.data[i]);
                    let costs: DVector<f32> = NN::calculate_cost(&new_layers, &training_data_cloned.labels[i]);
                    let (delta_biases, delta_weights) = NN::backpropagation_for_parallelisation(&new_network, costs, &new_layers);

                    for i in &delta_biases {
                        if i.iter().find(|x| x.is_nan() ).is_some() { panic!("Got NaN biases") };
                    };

                    for i in &delta_weights {
                        if i.iter().find(|x| x.is_nan() ).is_some() { panic!("Got NaN weights") };
                    };

                    let mut guessed_correct = false;
                    if NN::network_classification(&new_layers[new_layers.len()-1]) == NN::network_classification(&training_data_cloned.labels[i]) { guessed_correct = true };
                    tx_cloned.send((delta_biases, delta_weights, guessed_correct)).unwrap()
                });
                handles.push(handle);
            }
            drop(tx);

            for handle in handles {
                let _ = handle.join().unwrap();
            }

            let first_value = rx.recv().unwrap();
            let mut delta_biases_sum = first_value.0;
            let mut delta_weights_sum = first_value.1;
            if first_value.2 { avg_score += 1.0 };
            for recieved in rx {
                if recieved.2 { avg_score += 1.0 };
                delta_biases_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&recieved.0[i]);
                delta_weights_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&recieved.1[i]);
            }

            //let to_apply_weights: Vec<DMatrix<f32>> = delta_weights_sum.iter().map(|x| x * 1.0/(avg_score*0.5) * (1.0/(cycle_size as f32))).collect();
            //let to_apply_biases: Vec<DVector<f32>> = delta_biases_sum.iter().map(|x| x * 1.0/(avg_score*0.5) * (1.0/(cycle_size as f32))).collect();
            let to_apply_weights: Vec<DMatrix<f32>> = delta_weights_sum.iter().map(|x| x * 0.1 * (1.0/(cycle_size as f32))).collect();
            let to_apply_biases: Vec<DVector<f32>> = delta_biases_sum.iter().map(|x| x * 0.1 * (1.0/(cycle_size as f32))).collect();


            network.weights.iter_mut().enumerate().for_each(|(i,x)| *x -= &to_apply_weights[i]);
            network.biases.iter_mut().enumerate().for_each(|(i,x)| *x -= &to_apply_biases[i]);

            avg_score = avg_score/(cycle_size as f32);
            println!("{}", avg_score);
            if iterator_over_cycles + cycle_size > training_data_ref.data.len() {
                iterator_over_cycles = cycle_size;
            } else {
                iterator_over_cycles += cycle_size;
            }
        }
        network
    }


    fn non_parallel_training (mut network: NN, cycle_size: usize, training_data: TrainingData, precision: f32) -> NN {
        assert!(cycle_size<training_data.data.len());
        let mut avg_cost: f32 = 0.0;
        let mut iterator_over_cycles = cycle_size;

        while avg_cost < precision {
            avg_cost = 0.0;
            network.layers = NN::forward_pass(&network, &training_data.data[j]);
            let costs: DVector<f32> = NN::calculate_cost(&network.layers, &training_data.labels[j]);
            let (mut delta_biases_sum, mut delta_weights_sum) = NN::backpropagation(&network, costs);
            for i in iterator_over_cycles-cycle_size+1..(iterator_over_cycles) {
                network.layers = NN::forward_pass(&network, &training_data.data[i]);
                let costs: DVector<f32> = NN::calculate_cost(&network.layers, &training_data.labels[i]);
                let (delta_biases, delta_weights) = NN::backpropagation(&network, costs);

                for i in &delta_biases {
                    if i.iter().find(|x| x.is_nan() ).is_some() { panic!("Got NaN biases") };
                };

                for i in &delta_weights {
                    if i.iter().find(|x| x.is_nan() ).is_some() { panic!("Got NaN weights") };
                };

                delta_biases_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&delta_biases[i]);
                delta_weights_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&delta_weights[i]);


                if NN::network_classification(&network.layers[network.layers.len()-1]) == NN::network_classification(&training_data.labels[i]) { avg_cost += 1.0 };
            }

            // let to_apply_weights: Vec<DMatrix<f32>> = delta_weights_sum.iter().map(|x| x * 1.0/(avg_cost*0.5) * (1.0/(cycle_size as f32))).collect();
            // let to_apply_biases: Vec<DVector<f32>> = delta_biases_sum.iter().map(|x| x * 1.0/(avg_cost*0.5) * (1.0/(cycle_size as f32))).collect();
            let to_apply_weights: Vec<DMatrix<f32>> = delta_weights_sum.iter().map(|x| x * 0.1 * (1.0/(cycle_size as f32))).collect();
            let to_apply_biases: Vec<DVector<f32>> = delta_biases_sum.iter().map(|x| x * 0.1 * (1.0/(cycle_size as f32))).collect();


            network.weights.iter_mut().enumerate().for_each(|(i,x)| *x -= &to_apply_weights[i]);
            network.biases.iter_mut().enumerate().for_each(|(i,x)| *x -= &to_apply_biases[i]);
            avg_cost = avg_cost/(cycle_size as f32);

            println!("{}", avg_cost);
            if iterator_over_cycles + cycle_size > training_data.data.len() {
                iterator_over_cycles = cycle_size;
            } else {
                iterator_over_cycles += cycle_size;
            }
        }
        network
    }
}
