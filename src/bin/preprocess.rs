use byteorder::{BigEndian, ReadBytesExt};
use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufWriter;
use std::path::Path;

// TODO: Try to use the mnist crate here
fn main() {
    eprintln!("Converting mninst binary train data to fann format...");
    convert_mnist_to_fann(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
        "train.fann",
    )
    .map_err(|err| format!("Failed to convert train images and labels: {}", err))
    .unwrap();
    eprintln!("Ok");

    eprintln!("Converting mninst binary test data to fann format...");
    convert_mnist_to_fann(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
        "test.fann",
    )
    .map_err(|err| format!("Failed to convert t10k images and labels: {}", err))
    .unwrap();
    eprintln!("Ok");
}

fn convert_mnist_to_fann<P: AsRef<Path>>(
    images: P,
    labels: P,
    output: P,
) -> Result<(), Box<dyn Error>> {
    let mut images_file = OpenOptions::new().read(true).open(images)?;
    let mut labels_file = OpenOptions::new().read(true).open(labels)?;
    let output_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(output)?;

    let mut output_file = BufWriter::new(output_file);

    // we don't care about the first 4 bytes (magic number)
    images_file.read_u32::<BigEndian>()?;

    // read the number of samples
    let num_samples = images_file.read_u32::<BigEndian>()?;

    // read the number of cols and rows per image
    let num_cols = images_file.read_u32::<BigEndian>()?;
    let num_rows = images_file.read_u32::<BigEndian>()?;

    // discard the first 8 bytes of the label file (magic number and samples count)
    labels_file.read_u64::<BigEndian>()?;

    // write the header for the fann output file
    write!(
        output_file,
        "{} {} {} \n",
        num_samples,
        num_cols * num_rows,
        10
    )?;

    for _ in 1..=num_samples {
        // get the label for this sample
        let label = labels_file.read_u8()?;
        // create an one hot encode placeholder
        let mut labels = vec![0.1; 10];
        // set the correct label with a higher value
        labels[label as usize] = 0.9;

        // loop through all pixel values in a sample
        for _ in 1..=(num_cols * num_rows) {
            let pixel = images_file.read_u8()?;
            write!(output_file, "{} ", pixel as f32 / 255.)?;
        }

        write!(output_file, "\n")?;
        for l in &labels {
            write!(output_file, "{} ", l)?;
        }
        write!(output_file, "\n")?;
    }

    Ok(())
}
