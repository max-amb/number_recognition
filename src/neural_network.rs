use nalgebra::{DMatrix, DVector};
use rand::Rng; use rand_distr::{Distribution, Normal};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::sync::{mpsc, Arc};
use std::thread;

use crate::training_data::TrainingData;
use crate::optimisation_algos::{OptimisationAlgorithms, Optimisation};
use crate::activation_functions;

#[derive(PartialEq, Default)]
pub enum InitialisationOptions {
    Random,
    #[default] He,
}

#[derive(Debug)]
pub enum CostFunction {
    Quadratic,
    CategoricalCrossEntropy,
}

impl CostFunction {
    pub fn calculate_cost (&self, observed: &DVector<f32>, expected: &DVector<f32>) -> f32 {
        match self {
            CostFunction::Quadratic => {
                observed.iter().enumerate().map(|(i, x)| (x-expected[i]).powi(2)).sum::<f32>()
            },
            CostFunction::CategoricalCrossEntropy => {
                -expected.iter().enumerate().map(|(i, x)| x*((f32::EPSILON+observed[i]).ln())).sum::<f32>()
            },
        }
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
                    let normal_dist = Normal::new(0.0, (2.0_f32/(layer_sizes[(x-1) as usize] as f32)).sqrt()).unwrap();
                    normal_dist.sample(&mut rng) }
                )) 
                .collect() },
        };

        let biases: Vec<DVector<f32>> = (1..number_of_layers)
            .map(|x| DVector::from_fn(layer_sizes[x as usize] as usize, |_, _| rng.random_range(-1.0..=1.0)))
            .collect();

        let alpha = alpha_value.unwrap_or(0.01);
        Ok(Self {
            layers,
            weights,
            biases,
            alpha,
        })
    }

    pub fn forward_pass (network: &NN, input: &DVector<f32>, cost_function: &CostFunction) -> Vec<DVector<f32>> {
        let mut new_layers: Vec<DVector<f32>> = vec![ input.clone() ];
        for layer in 0..network.weights.len()-1 {
            new_layers.push((&network.weights[layer]*&new_layers[layer] + &network.biases[layer]).map(|x| activation_functions::leaky_relu(x, network.alpha)));
        }
        new_layers.push(match cost_function {
            CostFunction::Quadratic => (&network.weights[network.weights.len()-1]*&new_layers[network.weights.len()-1] + &network.biases[network.weights.len()-1]).map(activation_functions::sigmoid),
            CostFunction::CategoricalCrossEntropy => {
                activation_functions::softmax(&network.weights[network.weights.len()-1]*&new_layers[network.weights.len()-1] + &network.biases[network.weights.len()-1])
            },
        });
        new_layers
    }

    pub fn backprop (network: &NN, expected_result: &DVector<f32>, new_layers: &[DVector<f32>], cost_function: &CostFunction) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
        let mut delta_weights_list: Vec<DMatrix<f32>> = Vec::new(); 
        let mut delta_biases_list: Vec<DVector<f32>> = Vec::new(); 

        for layer in (0..network.weights.len()).rev() { // Layers
            let non_activation_applied_layer = &network.weights[layer]*&new_layers[layer]+&network.biases[layer];

            if layer == network.weights.len()-1 {
                delta_biases_list.push(match cost_function {
                    CostFunction::Quadratic => DVector::from_iterator(network.layers[layer+1].nrows(),
                        (2.0 * (&new_layers[layer+1] - expected_result))
                        .iter().enumerate()
                        .map(|(i,x)| x*new_layers[layer+1][i]*(1.0 - &new_layers[layer+1][i]))),
                    CostFunction::CategoricalCrossEntropy => &new_layers[layer+1] - expected_result,
                });
            } else {
                delta_biases_list.push(DVector::from_iterator( network.layers[layer+1].nrows(),
                    (network.weights[layer+1].transpose() * delta_biases_list.last().unwrap())
                        .iter().enumerate()
                        .map(|(i,x)| x*activation_functions::leaky_relu_derivative(non_activation_applied_layer[i], network.alpha))
                ));
            }
            delta_weights_list.push(delta_biases_list.last().unwrap()*(new_layers[layer].transpose())); 
        }

        (delta_biases_list.into_iter().rev().collect(), delta_weights_list.into_iter().rev().collect())
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
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut output: String = String::from("[layers]\n");
        output.push_str(&network.layers.iter().fold(String::from("["), |s, x| format!("{s}{}, ", x.len())));
        output.replace_range(output.len()-2..output.len(), "]\n");

        output.push_str("\n[weights]");
        for weight in network.weights.iter().enumerate() {
            output.push_str("\n[");
            for individual_number in weight.1 {
                output.push_str(&format!("{individual_number}, "));
            }
            output.replace_range(output.len()-2..output.len(), "]");
        }

        output.push_str("\n\n[biases]");
        for biase in network.biases.iter().enumerate() {
            output.push_str("\n[");
            for individual_number in biase.1 {
                output.push_str(&format!("{individual_number}, "));
            }
            output.replace_range(output.len()-2..output.len(), "]");
        }

        output.push_str("\n\n[alpha]\n");
        output.push_str(&format!("{}\n", network.alpha));

        write!(f, "{output}")
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
            .map(|x| DVector::from_element(layer_sizes[x], 0.0))
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
        dbg!(&line);
        let alpha: f32 = line.strip_suffix("\n").unwrap().parse().unwrap();

        Ok(NN {
            layers,
            weights,
            biases,
            alpha
        })
    }

    pub fn training (
        mut network: NN,
        cycle_size: usize,
        training_data: TrainingData,
        precision: f32,
        cost_function: CostFunction,
        optimisation_algorithm: OptimisationAlgorithms,
        learning_rate: f32)
    -> NN {

        assert!(cycle_size<=training_data.data.len());
        let mut optimisation = Optimisation::new(&network, optimisation_algorithm, learning_rate, cycle_size, Some(0.99));

        let training_data_ref = Arc::new(training_data);
        let cost_function_ref = Arc::new(cost_function);
        let mut avg_score: f32 = 0.0;
        let mut costs: f32 = 0.0;
        let mut iterator_over_cycles = 0;
        let mut epochs: u32 = 0;

        loop {
            let network_for_this_iter = Arc::new(network.clone());
            let (tx, rx) = mpsc::channel();
            let mut handles = vec![];

            for i in (iterator_over_cycles..iterator_over_cycles+cycle_size).map(|x| x%training_data_ref.data.len()) {
                let tx_cloned = tx.clone();
                let new_network = Arc::clone(&network_for_this_iter);
                let training_data_cloned = Arc::clone(&training_data_ref);
                let cost_function_cloned = Arc::clone(&cost_function_ref);

                let handle = thread::spawn(move || {
                    let new_layers = NN::forward_pass(&new_network, &training_data_cloned.data[i], &cost_function_cloned);
                    let (delta_biases, delta_weights) = NN::backprop(&new_network, &training_data_cloned.labels[i], &new_layers, &cost_function_cloned);

                    for i in &delta_biases {
                        // dbg!(i);
                        if i.iter().any(|x| x.is_nan()) {panic!("Got NaN biases") };
                    };

                    for i in &delta_weights {
                        if i.iter().any(|x| x.is_nan()) { panic!("Got NaN weights") };
                    };

                    let mut guessed_correct = false;
                    if NN::network_classification(&new_layers[new_layers.len()-1]) == NN::network_classification(&training_data_cloned.labels[i]) { guessed_correct = true };
                    tx_cloned.send((delta_biases, delta_weights, guessed_correct, new_layers[new_layers.len()-1].clone(), training_data_cloned.labels[i].clone())).unwrap()
                });
                handles.push(handle);
            }
            drop(tx);

            for handle in handles {
                handle.join().unwrap();
            }

            let first_value = rx.recv().unwrap();
            let mut delta_biases_sum = first_value.0;
            let mut delta_weights_sum = first_value.1;
            if first_value.2 { avg_score += 1.0 };
            costs += cost_function_ref.calculate_cost(&first_value.3, &first_value.4);
            for recieved in rx {
                if recieved.2 { avg_score += 1.0 };
                delta_biases_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&recieved.0[i]);
                delta_weights_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&recieved.1[i]);
                costs += cost_function_ref.calculate_cost(&recieved.3, &recieved.4);
            }

            let changes_to_apply: (Vec<DMatrix<f32>>, Vec<DVector<f32>>) = optimisation.calculate_change(delta_weights_sum, delta_biases_sum);

            network.weights.iter_mut().enumerate().for_each(|(i,x)| *x -= &changes_to_apply.0[i]);
            network.biases.iter_mut().enumerate().for_each(|(i,x)| *x -= &changes_to_apply.1[i]);

            if iterator_over_cycles > training_data_ref.data.len() {
                avg_score /= iterator_over_cycles as f32;
                costs /= iterator_over_cycles as f32;
                iterator_over_cycles-=training_data_ref.data.len();
                epochs += 1;
                println!();
                println!("epochs: {epochs}");
                println!("avg score: {avg_score}");
                println!("avg costs: {costs}");
                if avg_score > precision { break }
            }
            iterator_over_cycles += cycle_size;
        }
        network
    }


    pub fn non_parallel_training (mut network: NN, cycle_size: usize, training_data: TrainingData, precision: f32, cost_function: CostFunction, optimisation_algorithm: OptimisationAlgorithms, learning_rate: f32) -> NN {

        assert!(cycle_size<=training_data.data.len());
        let mut optimisation = Optimisation::new(&network, optimisation_algorithm, learning_rate, cycle_size, Some(0.99));
        let mut avg_score: f32 = 0.0;
        let mut iterator_over_cycles = 0;
        let mut epochs: u32 = 0;

        while avg_score < precision {
            let mut costs_vec: Vec<(DVector<f32>, DVector<f32>)> = Vec::new();
            avg_score = 0.0;
            let new_layers = NN::forward_pass(&network, &training_data.data[iterator_over_cycles%60000], &cost_function);
            let (mut delta_biases_sum, mut delta_weights_sum) = NN::backprop(&network, &training_data.labels[iterator_over_cycles%60000], &new_layers, &cost_function);
            if NN::network_classification(new_layers.last().unwrap()) == NN::network_classification(&training_data.labels[iterator_over_cycles%60000]) { avg_score += 1.0 };
            costs_vec.push((new_layers[new_layers.len()-1].clone(), training_data.labels[iterator_over_cycles%60000].clone()));

            for i in (iterator_over_cycles+1..cycle_size+iterator_over_cycles).map(|x| x%60000) {
                let new_layers = NN::forward_pass(&network, &training_data.data[i], &cost_function);
                let (delta_biases, delta_weights) = NN::backprop(&network, &training_data.labels[i], &new_layers, &cost_function);

                for i in &delta_biases {
                    if i.iter().any(|x| x.is_nan()) { panic!("Got NaN biases") };
                };

                for i in &delta_weights {
                    if i.iter().any(|x| x.is_nan()) { panic!("Got NaN weights") };
                };

                delta_biases_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&delta_biases[i]);
                delta_weights_sum.iter_mut().enumerate().for_each(|(i,x)| *x+=&delta_weights[i]);


                if NN::network_classification(new_layers.last().unwrap()) == NN::network_classification(&training_data.labels[i]) { avg_score += 1.0 };
                costs_vec.push((new_layers[new_layers.len()-1].clone(), training_data.labels[i].clone()));
            }

            let mut costs: f32 = costs_vec.iter().map(|x| cost_function.calculate_cost(&x.0, &x.1)).sum();
            

            let changes_to_apply: (Vec<DMatrix<f32>>, Vec<DVector<f32>>) = optimisation.calculate_change(delta_weights_sum, delta_biases_sum);

            network.weights.iter_mut().enumerate().for_each(|(i,x)| *x -= &changes_to_apply.0[i]);
            network.biases.iter_mut().enumerate().for_each(|(i,x)| *x -= &changes_to_apply.1[i]);

            avg_score /= cycle_size as f32;
            costs /= cycle_size as f32;

            if iterator_over_cycles > training_data.data.len() {
                iterator_over_cycles -= cycle_size;
                epochs += 1;
                println!();
                println!("epochs: {epochs}");
                println!("avg score: {avg_score}");
                println!("avg costs: {costs}");
            }
            iterator_over_cycles += cycle_size;
        }
        network
    }
}
