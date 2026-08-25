use nalgebra::DVector;
use std::fs::File;
use std::io::{BufReader, Read};

use crate::tensor::Ten;

pub struct TrainingData {
    pub data: Vec<Ten>,
    pub labels: Vec<Ten>,
}

impl TrainingData {
    pub fn new(
        file_path_of_labels: &str,
        file_path_of_images: &str,
        size_of_data: usize,
    ) -> TrainingData {
        let training = TrainingData {
            data: TrainingData::read_images(file_path_of_images, size_of_data).unwrap(),
            labels: TrainingData::read_labels(file_path_of_labels, size_of_data).unwrap(),
        };
        assert_eq!(training.labels.len(), training.data.len());
        training
    }

    pub fn read_images(
        file_path_of_images: &str,
        size_of_data: usize,
    ) -> Result<Vec<Ten>, std::io::Error> {
        let f = File::open(file_path_of_images)?;
        let mut reader = BufReader::with_capacity(4, f);
        let mut buffer = [0; 4];
        let mut images: Vec<Ten> = Vec::with_capacity(size_of_data);

        // Magic number
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(buffer[0], 0);
        assert_eq!(buffer[1], 0);
        assert_eq!(buffer[2], 8);
        assert_eq!(buffer[3], 3);

        // Number of items
        reader.read_exact(&mut buffer).unwrap();
        let number_of_images: u32 = buffer
            .iter()
            .rev()
            .enumerate()
            .map(|(i, x)| (*x as u32) * (256_u32.pow(i as u32)))
            .sum(); // Change to u32 to allow large numbers

        // Size of images
        reader.read_exact(&mut buffer).unwrap();
        let number_of_rows: usize = buffer
            .iter()
            .rev()
            .enumerate()
            .map(|(i, x)| (*x as usize) * (256_usize.pow(i as u32)))
            .sum(); // Change to u32 to allow large numbers

        reader.read_exact(&mut buffer).unwrap();
        let number_of_cols: usize = buffer
            .iter()
            .rev()
            .enumerate()
            .map(|(i, x)| (*x as usize) * (256_usize.pow(i as u32)))
            .sum(); // Change to u32 to allow large numbers

        // Reading images into vec
        for _ in 0..number_of_images {
            let mut buffer: Vec<u8> = vec![0u8; (number_of_rows * number_of_cols) as usize];
            reader.read_exact(&mut buffer).unwrap();
            images.push(Ten {
                data: DVector::from_iterator((number_of_rows * number_of_cols) as usize,buffer.into_iter().map(|x| (x as f32) / 255.0_f32)),
                shape: (number_of_rows, number_of_cols).into()
            });
        }

        Ok(images)
    }

    pub fn read_labels(
        file_path_of_labels: &str,
        size_of_data: usize,
    ) -> Result<Vec<Ten>, std::io::Error> {
        let f = File::open(file_path_of_labels)?;
        let mut reader = BufReader::with_capacity(4, f);
        let mut buffer = [0; 4];
        let mut labels: Vec<Ten> = Vec::with_capacity(size_of_data);

        // Magic number
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(buffer[0], 0);
        assert_eq!(buffer[1], 0);
        assert_eq!(buffer[2], 8);
        assert_eq!(buffer[3], 1);

        // Number of items
        reader.read_exact(&mut buffer).unwrap();
        let number_of_labels: u32 = buffer
            .iter()
            .rev()
            .enumerate()
            .map(|(i, x)| (*x as u32) * (256_u32.pow(i as u32)))
            .sum(); // Change to u32 to allow large numbers

        // Reading labels into vec
        let mut buffer = [0];
        for _ in 0..number_of_labels {
            reader.read_exact(&mut buffer).unwrap();
            labels.push(Ten { 
                data: DVector::from_fn(10, |i, _| {if i as u8 == buffer[0] { 1.0 } else { 0.0 }}),
                shape: (10, 1).into(),
            });
        }
        Ok(labels)
    }

    /*
    pub fn generate_training_data_from_bmp(
        image_path: &str,
    ) -> Result<Ten, std::io::Error> {
        let f = File::open(image_path)?;
        let mut reader = BufReader::with_capacity(256, f);

        // For header
        let mut buffer = [0; 14];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!([66_u8, 77_u8], buffer[0..2]); // BM
        let _size_of_file: u32 = buffer[2..6]
            .iter()
            .enumerate()
            .map(|(i, x)| (*x as u32) * (256_u32.pow(i as u32)))
            .sum(); // FROM MSB
        let _pixel_array_offset: u64 = buffer[10..14]
            .iter()
            .enumerate()
            .map(|(i, x)| (*x as u64) * (256_u64.pow(i as u32)))
            .sum();

        // DIB header
        let mut buffer = [0; 124];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!([124_u8, 0_u8, 0_u8, 0_u8], buffer[0..4]); // GIMP!! (v5)
        // see http://www.jose.it-berater.org/gdi/bitmaps/bitmapv5header.htm

        assert_eq!([28_u8, 0_u8, 0_u8, 0_u8], buffer[4..8]); // Width 28
        assert_eq!([28_u8, 0_u8, 0_u8, 0_u8], buffer[8..12]); // Length 28
        assert_eq!([1_u8, 0_u8], buffer[12..14]); // Planes (must be one)
        assert_eq!(
            8_u32,
            buffer[14..16]
                .iter()
                .enumerate()
                .map(|(i, x)| (*x as u32) * (256_u32.pow(i as u32)))
                .sum()
        ); // Bit count (bit width)
        assert_eq!([0_u8, 0_u8, 0_u8, 0_u8], buffer[16..20]); // Uncompressed

        let _size_of_image: u32 = buffer[20..24]
            .iter()
            .enumerate()
            .map(|(i, x)| (*x as u32) * (256_u32.pow(i as u32)))
            .sum();
        // [24..32] is scaling, not required
        let _colours_used: u32 = buffer[32..36]
            .iter()
            .enumerate()
            .map(|(i, x)| (*x as u32) * (256_u32.pow(i as u32)))
            .sum();
        let _colours_needed: u32 = buffer[36..40]
            .iter()
            .enumerate()
            .map(|(i, x)| (*x as u32) * (256_u32.pow(i as u32)))
            .sum();
        // [40..52] is not valid as no compression
        assert_eq!([0_u8, 0_u8, 0_u8, 0_u8], buffer[52..56]); // No alpha mask

        // Ignore rest of DIB

        // Pixel array
        reader.seek(SeekFrom::Start(_pixel_array_offset))?;
        let mut buffer = [0; 28 * 28]; // Width * Height with bit depth 8
        reader.read_exact(&mut buffer).unwrap();
        let x: Vec<f32> = buffer.iter().map(|x| (*x as f32) / 255.0).collect();
        let mut new: Vec<f32> = Vec::new();
        for i in x.iter().enumerate() {
            new.insert(i.0 % 28, *i.1);
        }
        Ok(DVector::from_vec(new))
    }*/
}
