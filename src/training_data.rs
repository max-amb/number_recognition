use nalgebra::DVector;
use std::io::{SeekFrom, Seek, BufReader, Read};
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

    pub fn generate_training_data_from_bmp (image_path: &str) -> Result<DVector<f32>, std::io::Error> {
        let f = File::open(image_path)?;
        let mut reader = BufReader::with_capacity(256, f);

        // For header
        let mut buffer = [0; 14];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!([66 as u8, 77 as u8], buffer[0..2]); // BM
        let _size_of_file: u32 = buffer[2..6].iter().enumerate().map(|(i, x)| (*x as u32)*(256_u32.pow(i as u32))).sum(); // FROM MSB
        let _pixel_array_offset: u64 = buffer[10..14].iter().enumerate().map(|(i, x)| (*x as u64)*(256_u64.pow(i as u32))).sum(); 

        // DIB header
        let mut buffer = [0; 124];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!([124 as u8, 0 as u8, 0 as u8, 0 as u8], buffer[0..4]); // GIMP!! (v5)
        // see http://www.jose.it-berater.org/gdi/bitmaps/bitmapv5header.htm

        assert_eq!([28 as u8, 0 as u8, 0 as u8, 0 as u8], buffer[4..8]); // Width 28
        assert_eq!([28 as u8, 0 as u8, 0 as u8, 0 as u8], buffer[8..12]); // Length 28
        assert_eq!([1 as u8, 0 as u8], buffer[12..14]); // Planes (must be one)
        assert_eq!(8 as u32, buffer[14..16].iter().enumerate().map(|(i, x)| (*x as u32)*(256_u32.pow(i as u32))).sum()); // Bit count (bit width)
        assert_eq!([0 as u8, 0 as u8, 0 as u8, 0 as u8], buffer[16..20]); // Uncompressed
        
        let _size_of_image: u32 = buffer[20..24].iter().enumerate().map(|(i, x)| (*x as u32)*(256_u32.pow(i as u32))).sum(); 
        // [24..32] is scaling, not required
        let _colours_used: u32 = buffer[32..36].iter().enumerate().map(|(i, x)| (*x as u32)*(256_u32.pow(i as u32))).sum();
        let _colours_needed: u32 = buffer[36..40].iter().enumerate().map(|(i, x)| (*x as u32)*(256_u32.pow(i as u32))).sum();
        // [40..52] is not valid as no compression
        assert_eq!([0 as u8, 0 as u8, 0 as u8, 0 as u8], buffer[52..56]); // No alpha mask

        // Ignore rest of DIB
        
        // Pixel array
        reader.seek(SeekFrom::Start(_pixel_array_offset))?;
        let mut buffer = [0; 28*28]; // Width * Height with bit depth 8
        reader.read_exact(&mut buffer).unwrap();
        let x: Vec<f32> = buffer.iter().map(|x| (*x as f32)/255.0).collect();
        let mut new: Vec<f32> = Vec::new();
        for i in x.iter().enumerate() {
            new.insert(i.0%28, *i.1);
        }
        Ok(DVector::from_vec(new))
    }

    #[allow(dead_code)]
    fn crc32_for_png (_data: &[u8]) -> bool {
        // TODO!!
        true
    }

    #[allow(dead_code)]
    fn generate_training_data_from_png (image_path: &str) -> Result<TrainingData, std::io::Error> {
        // https://en.wikipedia.org/wiki/PNG
        let f = File::open(image_path)?;
        let mut reader = BufReader::with_capacity(128, f);
        let mut buffer = [0; 8];

        // Header
        reader.read(&mut buffer).unwrap();
        assert_eq!([137 as u8,80 as u8,78 as u8,71 as u8,13 as u8,10 as u8,26 as u8,10 as u8], buffer); // Assert that the image is a PNG

        // IHDR chunk
        let mut buffer = [0; 4];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!([0 as u8, 0 as u8, 0 as u8, 13 as u8], buffer); // Assert the next chunk (IHDR)
        // has length 13

        let mut buffer = [0; 4];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!([73 as u8, 72 as u8, 68 as u8, 82 as u8], buffer); // Assert the chunks name is
        // IHDR

        let mut buffer = [0; 17];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!([0 as u8, 0 as u8, 0 as u8, 28 as u8], buffer[0..4]); // Width of 28 
        assert_eq!([0 as u8, 0 as u8, 0 as u8, 28 as u8], buffer[4..8]); // Length of 28 
        let _bit_depth = buffer[8..9][0]; // Bits per pixel
        assert_eq!([0 as u8], buffer[9..10]); // Assert greyscale
        assert_eq!([0 as u8], buffer[10..11]); // Compression 
        assert_eq!([0 as u8], buffer[11..12]); // Filtering
        assert_eq!([0 as u8], buffer[12..13]); // Interlacing
        if !TrainingData::crc32_for_png(&buffer) { panic!() }; // Check for checksum

        let mut buffer = [0; 4];
        reader.read_exact(&mut buffer).unwrap();
        let _length_of_data: u32 = buffer.iter().rev().enumerate().map(|(i, x)| (*x as u32)*(256_u32.pow(i as u32))).sum();

        reader.read_exact(&mut buffer).unwrap();
        assert_eq!([73 as u8, 68 as u8, 65 as u8, 84 as u8], buffer); // Check it is the IDAT data

        // see https://datatracker.ietf.org/doc/html/rfc1950#section-2

        let mut buffer = [0; 1];
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!([120 as u8], buffer); // Assert CMF is 0x78 - 7 for 32K window size and 8 for
        // compression type deflate

        reader.read_exact(&mut buffer).unwrap();
        assert!((120*256+(buffer[0] as u32)) % 31 == 0);  // Check digits (FCHECK)
        assert!(buffer[0] & 8 == 0); // Assert no dictionary (FDICT)
        // We can safely ignore the rest of the byte as FLEVEL isn't needed for decomp
     


        // Calculating the amount of data to be read

        Ok (TrainingData {
            data: TrainingData::read_images("").unwrap(),
            labels: TrainingData::read_labels("").unwrap(),
        })
    }
}

