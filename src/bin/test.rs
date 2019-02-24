use fann::{Fann};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<(), Box<dyn Error>> {
    let mut features = vec![];
    let mut errors = 0;
    let mut correct = 0;

    let fann = Fann::from_file("checkpoints/mnist.net")?;

    let f = File::open("test.fann")?;
    let f = BufReader::new(&f);
    for line in f.lines() {
        let line = line.unwrap();
        let pieces: Vec<f32> = line.trim().split(' ').map(|s| s.parse().unwrap()).collect();

        let is_input = pieces.len() as u32 == fann.get_num_input();
        let is_output = pieces.len() as u32 == fann.get_num_output();

        if is_input {
            features = pieces; 
            continue;
        } else if is_output {
            let output = fann.run(features.as_slice()).unwrap();
            let output_confidence = output.iter().fold(0.0, |acc: f32, &x| acc.max(x));
            let output_prediction = output.iter().position(|&x| x == output_confidence).unwrap();
            
            let true_label_confidence = pieces.iter().fold(0.0, |acc: f32, &x| acc.max(x));
            let true_label = pieces.iter().position(|&x| x == true_label_confidence).unwrap();

            eprintln!("I think this number is {} with {:.2}% confidence", output_prediction, output_confidence * 100.0);
            eprintln!("REAL VALUE: {}", true_label);

            if output_prediction != true_label {
                errors+=1;
            } else {
                correct+=1;
            }
        }
    }

    let total = errors + correct;

    eprintln!("Total samples: {}", total);
    eprintln!("Correct: {}", correct);
    eprintln!("Errors: {}", errors);
    eprintln!("Accuracy: {:.2}%", correct as f32 * 100.0 / total as f32);

    Ok(())
}
