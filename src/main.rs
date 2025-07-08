use nalgebra::{DMatrix, DVector};
use rand::random;
use std::cmp::max;
use std::io::{BufReader, Read};
use std::fs::File;
use image::GrayImage;

#[derive(Debug)]
struct NN {
    layers: Vec<DVector<f32>>,
    weights: Vec<DMatrix<f32>>,
    biases: Vec<DVector<f32>>,
}

impl NN {
    fn new (number_of_layers: i32, layer_sizes: &[i32]) -> Result<NN, &'static str> {
        if layer_sizes.len() != number_of_layers as usize {
            return Err("The array of layer sizes has a different number of elements than the number of layers");
        } 

        let layers: Vec<DVector<f32>> = (0..number_of_layers)
            .map(|x| DVector::from_element(layer_sizes[x as usize] as usize, 0.0))
            .collect();
        let weights: Vec<DMatrix<f32>> = (1..number_of_layers)
            .map(|x| DMatrix::from_fn(layer_sizes[x as usize] as usize, layer_sizes[(x-1) as usize] as usize, |_, _| random()))
            .collect();
        let biases: Vec<DVector<f32>> = (1..number_of_layers)
            .map(|x| DVector::from_fn(layer_sizes[x as usize] as usize, |_, _| random()))
            .collect();
        return Ok(Self {
            layers: layers,
            weights: weights,
            biases: biases,
        })
    }

    fn forward_pass (network: &NN) -> Vec<DVector<f32>> {
        let mut new_layers: Vec<DVector<f32>> = vec![ network.layers[0].clone() ];
        for layer in 0..network.weights.len() {
            new_layers.push((&network.weights[layer]*&network.layers[layer] + &network.biases[layer]).map(|x| NN::relu(x)));
        }
        new_layers
    }

    fn backpropagation (network: NN, expected_result: DVector<f32>, input: DVector<f32>) {}

    fn relu (input: f32) -> f32 {
        if input.lt(&0.0) {
            0.0
        } else {
            input
        }
    }

    fn relu_derivative (input: f32) -> f32 {
        if input.lt(&0.0) {
            0.0
        } else {
            1.0
        }
    }
}

fn main() {
    let mut network = match NN::new(4, &[784, 16, 16, 10]) {
        Ok(network) => network,
        Err(e) => { print!("{e:?}"); panic!("noooo") },
    };
    network.layers = NN::forward_pass(&network);

    let training_data = Training::new("/home/max/Downloads/train-labels.idx1-ubyte", "/home/max/Downloads/train-images.idx3-ubyte");
}

struct Training{
    data: Vec<DVector<u8>>,
    labels: Vec<DVector<u8>>,
}

impl Training {
    fn new (file_path_of_labels: &str, file_path_of_images: &str) -> Training {
        let training = Training {
            data: Training::read_images(file_path_of_images).unwrap(),
            labels: Training::read_labels(file_path_of_labels).unwrap(),
        };
        assert_eq!(training.labels.len(), training.data.len());
        training
    }

    fn read_images (file_path_of_images: &str) -> Result<Vec<DVector<u8>>, std::io::Error> {
        let f = File::open(file_path_of_images)?;
        let mut reader = BufReader::with_capacity(4, f);
        let mut buffer = [0; 4];
        let mut images: Vec<DVector<u8>> = Vec::new();
        
        // Magic number
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(buffer[2], 8);
        assert_eq!(buffer[3], 3);

        // Number of items
        reader.read_exact(&mut buffer).unwrap();
        let number_of_images: u32 = buffer.iter().rev().enumerate().map(|(i, x)| (*x as u32)*(256_u32.pow(i as u32))).sum(); // Change to u32 to allow large numbers
        
        // Size of images
        reader.read_exact(&mut buffer).unwrap();
        let number_of_rows: u32 = buffer.iter().rev().enumerate().map(|(i, x)| (*x as u32)*(256_u32.pow(i as u32))).sum(); // Change to u32 to allow large numbers

        reader.read_exact(&mut buffer).unwrap();
        let number_of_cols: u32 = buffer.iter().rev().enumerate().map(|(i, x)| (*x as u32)*(256_u32.pow(i as u32))).sum(); // Change to u32 to allow large numbers

        // Reading images into vec
        for _ in 0..number_of_images {
            let mut buffer: Vec<u8> = vec![0u8; (number_of_rows*number_of_cols) as usize];
            reader.read_exact(&mut buffer).unwrap();
            images.push(DVector::from_vec(buffer));
        };

        Ok(images)
    }

    #[allow(dead_code)]
    fn view_image (image: Vec<u8>) {
        let img = GrayImage::from_raw(28, 28, image).unwrap();
        img.save("./image.png").unwrap();
    }

    fn read_labels (file_path_of_labels: &str) -> Result<Vec<DVector<u8>>, std::io::Error> {
        let f = File::open(file_path_of_labels)?;
        let mut reader = BufReader::with_capacity(4, f); 
        let mut buffer = [0; 4];
        let mut labels: Vec<DVector<u8>> = Vec::new(); 

        // Magic number
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(buffer[2], 8);
        assert_eq!(buffer[3], 1);

        // Number of items
        reader.read_exact(&mut buffer).unwrap();
        let number_of_labels: u32 = buffer.iter().rev().enumerate().map(|(i, x)| (*x as u32)*(256_u32.pow(i as u32))).sum(); // Change to u32 to allow large numbers

        // Reading labels into vec
        let mut buffer = [0]; 
        for _ in 0..number_of_labels {
            reader.read_exact(&mut buffer).unwrap();
            labels.push(DVector::from_fn(9,
                |i, _| {if ((i+1) as u8) == buffer[0] {1} else {0}}));
        }

        Ok(labels)
    }
}
