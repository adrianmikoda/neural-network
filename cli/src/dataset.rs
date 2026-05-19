#![allow(dead_code)]

use std::fs::File;
use std::io::{BufReader, Error, ErrorKind, Read, Result};
use std::path::Path;

pub struct Dataset {
    pub images: Vec<Vec<f32>>,
    pub labels: Vec<u8>,
}

impl Dataset {
    pub fn load<P: AsRef<Path>>(images_path: P, labels_path: P) -> Result<Self> {
        let images = Self::read_images(images_path)?;
        let labels = Self::read_labels(labels_path)?;

        if images.len() != labels.len() {
            return Err(Error::new(ErrorKind::InvalidData, "Files don't match up"));
        }

        Ok(Self { images, labels })
    }

    fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_be_bytes(buf))
    }

    fn read_images<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<f32>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let magic = Self::read_u32(&mut reader)?;
        if magic != 2051 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Invalid MNIST magic number",
            ));
        }

        let num_images = Self::read_u32(&mut reader)? as usize;
        let rows = Self::read_u32(&mut reader)? as usize;
        let cols = Self::read_u32(&mut reader)? as usize;
        let image_size = rows * cols;

        let mut buf = vec![0u8; num_images * image_size];
        reader.read_exact(&mut buf)?;

        let mut images = Vec::with_capacity(num_images);
        for i in 0..num_images {
            let start = i * image_size;
            let end = start + image_size;

            let image = buf[start..end].iter().map(|&b| b as f32 / 255.0).collect();

            images.push(image);
        }

        Ok(images)
    }

    fn read_labels<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let magic = Self::read_u32(&mut reader)?;
        if magic != 2049 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Invalid MNIST magic number",
            ));
        }

        let num_labels = Self::read_u32(&mut reader)? as usize;

        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels)?;

        Ok(labels)
    }
}
