use std::cmp::min;

use crate::cost::CostFunction;
use crate::layers::{Activation, Backward, Delta, Forward, Initialisable, Layer};
use crate::tensor::{Shape, Ten};
use crate::training_data::TrainingData;

#[derive(Debug)]
pub struct NN {
    pub layers: Vec<Layer>,
    pub cost_function: CostFunction,
}

impl NN {
    pub fn new(mut layers: Vec<Layer>, input_shape: Shape, cost_function: CostFunction) -> NN {
        let mut curr_shape = input_shape;
        layers
            .iter_mut()
            .for_each(|x| curr_shape = x.initialise(curr_shape));
        NN {
            layers,
            cost_function,
        }
    }

    pub fn forward_pass(network: &NN, input: &Ten) -> Vec<Ten> {
        let mut new_layers: Vec<Ten> = vec![input.clone()];
        let mut curr_inp = input;
        for layer in &network.layers {
            let new_inp = layer.run(curr_inp);
            new_layers.push(new_inp);
            curr_inp = &new_layers[new_layers.len()-1];
        }
        let pos = new_layers.len()-1;
        new_layers[pos] = Activation::Softmax.run(&new_layers[pos]);
        new_layers
    }

    pub fn backprop(network: &NN, resultant_layers: Vec<Ten>, expected_result: &Ten) -> Option<Vec<Delta>> {
        let mut deltas: Vec<Delta> = Vec::new();
        let mut r = resultant_layers.into_iter();
        let softmax_result = r.next_back()?;
        let layer_iter = std::iter::zip(&network.layers, r).rev();
        let mut current_derivative = &softmax_result-expected_result;
        for (layer, layer_data) in layer_iter {
            let results = layer.backprop(current_derivative, &layer_data);
            current_derivative = results.0;
            deltas.push(results.1);
        }
        Some(deltas)
    }

    fn training_step<I>(network: &NN, run_iterator: I, training_data: &TrainingData) -> impl Iterator<Item = Delta> + use<I>
    where 
        I: IntoIterator<Item = usize>
    {
        let mut deltas: Vec<Delta> = Vec::new();
        let mut count = 0;
        for run_num in run_iterator {
            let tdm = &training_data.data[run_num];
            let tdl = &training_data.labels[run_num];
            let result = Self::forward_pass(network, &tdm);
            if NN::network_classification(&result[result.len()-1]) == NN::network_classification(tdl) { count += 1; }
            let ds = Self::backprop(network, result, tdl).unwrap();
            if deltas.is_empty() {
                deltas = ds
            } else {
                deltas = std::iter::zip(deltas, ds).map(|(d1, d2)| d1 + d2).collect();
            }
        }
        dbg!(count);
        deltas.into_iter().rev()
    }

    pub fn train(network: &mut NN, cycle_size: usize, training_data: TrainingData, precision: f32, learning_rate: f32) {
        let recip_cycle_size = 1.0/(cycle_size as f32);
        loop {
            for cycle in (0..training_data.data.len()).step_by(cycle_size) {
                dbg!(cycle);
                let deltas = Self::training_step(network, cycle..min(cycle+cycle_size, training_data.data.len()), &training_data);
                for (d, l) in std::iter::zip(deltas, &mut network.layers) {
                    l.apply(d*learning_rate*recip_cycle_size) 
                };
                break
            }
            // dbg!(Self::run_on_testing_data(network));
        }
    }

    /*
    #[allow(clippy::too_many_arguments)]
    pub fn training(
        mut network: NN,
        cycle_size: usize,
        training_data: TrainingData,
        precision: f32,
        cost_function: CostFunction,
        optimisation_algorithm: OptimisationAlgorithms,
        learning_rate: f32,
        step_size: usize
    ) -> NN {
        assert!(cycle_size <= training_data.data.len());
        assert!(step_size <= training_data.data.len());

        let mut optimisation = Optimisation::new(
            &network,
            optimisation_algorithm,
            learning_rate,
            cycle_size,
            Some(0.99),
        );

        let (ctrlc_transmitter, ctrlc_reciever) = mpsc::channel();
        ctrlc::set_handler(move || {
            ctrlc_transmitter
                .send(())
                .expect("Could not send signal on channel.")
        })
        .expect("Error setting Ctrl-C handler");

        let training_data_ref = Arc::new(training_data);
        let cost_function_ref = Arc::new(cost_function);
        let mut iterator_over_cycles = 0;

        while ctrlc_reciever.try_recv().is_err() {
            let network_for_this_iter = Arc::new(network.clone());
            let (tx, rx) = mpsc::channel();
            let mut handles = vec![];

            for i in (iterator_over_cycles..iterator_over_cycles + cycle_size).step_by(step_size)
            {
                let tx_cloned = tx.clone();
                let new_network = Arc::clone(&network_for_this_iter);
                let training_data_cloned = Arc::clone(&training_data_ref);
                let cost_function_cloned = Arc::clone(&cost_function_ref);

                let handle = thread::spawn(move || {
                    let mut deltas = Vec::new();
                    for j in (i..i+step_size).map(|x| x%training_data_cloned.data.len()){
                        let new_layers = NN::forward_pass(
                            &new_network,
                            &training_data_cloned.data[j],
                            &cost_function_cloned,
                        );
                        deltas.push(NN::backprop(
                            &new_network,
                            &training_data_cloned.labels[j],
                            &new_layers,
                            &cost_function_cloned,
                        ));
                    };

                    let mut delta_biases = deltas[0].0.clone();
                    let mut delta_weights= deltas[0].1.clone();
                    for bias_weight_pair in deltas.iter().skip(1) {
                        delta_biases.iter_mut().enumerate().for_each(|(j, x)| *x += &bias_weight_pair.0[j]);
                        delta_weights.iter_mut().enumerate().for_each(|(j, x)| *x += &bias_weight_pair.1[j]);
                    }

                    tx_cloned
                        .send((
                            delta_biases,
                            delta_weights,
                        ))
                        .unwrap()
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
            for recieved in rx { // Assumes all data has been recieved
                delta_biases_sum
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x += &recieved.0[i]);
                delta_weights_sum
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x += &recieved.1[i]);
            }

            let changes_to_apply: (Vec<DMatrix<f32>>, Vec<DVector<f32>>) =
                optimisation.calculate_change(&delta_weights_sum, &delta_biases_sum);

            network
                .weights
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x -= &changes_to_apply.0[i]);
            network
                .biases
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x -= &changes_to_apply.1[i]);

            println!("{iterator_over_cycles}");
            if iterator_over_cycles >= training_data_ref.data.len() {
                iterator_over_cycles = 0;
                let testing_data_score = NN::run_on_testing_data(&network, &cost_function_ref);
                let x: f32 = (testing_data_score as f32) / 10000.0;
                println!("{x}");
                if x > precision {
                    break;
                }
            }
            iterator_over_cycles += cycle_size;
        }

        ctrlc_reciever
            .recv()
            .expect("Didn't recieve ctrl-c signal from channel");
        NN::output_model_to_file(
            &network,
        "/home/max/projects/number_recognition/models/tmp.txt",
        ).unwrap();

        network
    }*/

    /*
    pub fn non_parallel_training(
        mut network: NN,
        cycle_size: usize,
        training_data: TrainingData,
        precision: f32,
        cost_function: CostFunction,
        optimisation_algorithm: OptimisationAlgorithms,
        learning_rate: f32,
    ) -> NN {
        assert!(cycle_size <= training_data.data.len());
        /*
        let mut avg_score: f32 = 0.0;
        let mut epochs: u32 = 0; */
        let mut iterator_over_cycles = 0;

        loop {
            let new_layers = NN::forward_pass(
                &network,
                &training_data.data[iterator_over_cycles % 60000],
                &cost_function,
            );
            let (mut delta_biases_sum, mut delta_weights_sum) = NN::backprop(
                &network,
                &training_data.labels[iterator_over_cycles % 60000],
                &new_layers,
                &cost_function,
            );

            for i in
                (iterator_over_cycles + 1..cycle_size + iterator_over_cycles).map(|x| x % 60000)
            {
                let new_layers = NN::forward_pass(&network, &training_data.data[i], &cost_function);
                let (delta_biases, delta_weights) = NN::backprop(
                    &network,
                    &training_data.labels[i],
                    &new_layers,
                    &cost_function,
                );

                delta_biases_sum
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x += &delta_biases[i]);
                delta_weights_sum
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x += &delta_weights[i]);
            }

            let changes_to_apply: (Vec<DMatrix<f32>>, Vec<DVector<f32>>) =
                optimisation.calculate_change(&delta_weights_sum, &delta_biases_sum);

            network
                .weights
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x -= &changes_to_apply.0[i]);
            network
                .biases
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x -= &changes_to_apply.1[i]);

            if iterator_over_cycles > training_data.data.len() {
                iterator_over_cycles = 0;
                let testing_data_score = NN::run_on_testing_data(&network, &cost_function);
                let x: f32 = (testing_data_score as f32) / 10000.0;
                println!("{x}");
                if x > precision {
                    break;
                }
            }
            iterator_over_cycles += cycle_size;
        }
        network
    }

    #[allow(dead_code)]
    pub fn run_on_testing_data(network: &NN, cost_function: &CostFunction) -> usize {
        let testing_data = TrainingData::new(
            "/home/max/Downloads/t10k-labels.idx1-ubyte",
            "/home/max/Downloads/t10k-images.idx3-ubyte",
            60000,
        );
        let mut correct: usize = 0;
        for j in 0..testing_data.data.len() {
            let new_layers = NN::forward_pass(network, &testing_data.data[j], cost_function);
            if NN::network_classification(&new_layers[network.layers.len() - 1])
                == NN::network_classification(&testing_data.labels[j])
            {
                correct += 1
            };
        }
        correct
    }
    */

    pub fn network_classification(layer: &Ten) -> usize {
        let mut network_classification: (usize, f32) = (usize::MIN, f32::MIN);
        for i in layer.data.iter().enumerate() {
            if network_classification.1 < *i.1 {
                network_classification.1 = *i.1;
                network_classification.0 = i.0
            };
        }
        network_classification.0
    }

    pub fn run_on_testing_data(network: &NN) -> usize {
        let testing_data = TrainingData::new(
            "/home/max/Downloads/t10k-labels.idx1-ubyte",
            "/home/max/Downloads/t10k-images.idx3-ubyte",
            10000,
        );
        let mut correct: usize = 0;
        for j in 0..testing_data.data.len() {
            let new_layers = NN::forward_pass(network, &testing_data.data[j]);
            if NN::network_classification(&new_layers[new_layers.len() - 1])
                == NN::network_classification(&testing_data.labels[j])
            {
                correct += 1
            };
        }
        correct
    }
}
