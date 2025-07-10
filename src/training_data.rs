use nalgebra::DVector;
use std::io::{BufReader, Read};
use std::fs::File;
use image::GrayImage;

pub struct TrainingData{
    pub data: Vec<DVector<f32>>,
    pub labels: Vec<DVector<f32>>,
}

impl TrainingData {
    pub fn new (file_path_of_labels: &str, file_path_of_images: &str) -> TrainingData {
        let training = TrainingData {
            data: TrainingData::read_images(file_path_of_images).unwrap(),
            labels: TrainingData::read_labels(file_path_of_labels).unwrap(),
        };
        assert_eq!(training.labels.len(), training.data.len());
        training
    }
    
    pub fn read_images (file_path_of_images: &str) -> Result<Vec<DVector<f32>>, std::io::Error> {
        let f = File::open(file_path_of_images)?;
        let mut reader = BufReader::with_capacity(4, f);
        let mut buffer = [0; 4];
        let mut images: Vec<DVector<f32>> = Vec::new();
        
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
            images.push(DVector::from_iterator((number_of_rows*number_of_cols) as usize, buffer.into_iter().map(|x| (x as f32)/255.0_f32)));
        };
        
        Ok(images)
    }

    #[allow(dead_code)]
    fn view_image (images: &Vec<Vec<u8>>) {
        for image in images.iter().enumerate() {
            let img = GrayImage::from_raw(28, 28, image.1.clone()).unwrap();
            img.save(format!("/home/max/Pictures/images/image{}.png", image.0)).unwrap();
        }
    }

    pub fn read_labels (file_path_of_labels: &str) -> Result<Vec<DVector<f32>>, std::io::Error> {
        let f = File::open(file_path_of_labels)?;
        let mut reader = BufReader::with_capacity(4, f); 
        let mut buffer = [0; 4];
        let mut labels: Vec<DVector<f32>> = Vec::new(); 

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
            labels.push(DVector::from_fn(10,
                |i, _| {if ((i) as u8) == buffer[0] {1.0} else {0.0}}));
        }
        Ok(labels)
    }
}

