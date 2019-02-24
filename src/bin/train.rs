use std::error::Error;
use fann::{ActivationFunc, Fann, CallbackResult};

const LAYER_NEURONS: &[u32] = &[784, 256, 10];
const DESIRED_ERROR: f32 = 1e-3;
const MAX_EPOCHS: u32 = 150;

fn main() -> Result<(), Box<dyn Error>>{
   let mut fann = Fann::new(LAYER_NEURONS)?;

   fann.set_activation_func_hidden(ActivationFunc::Sigmoid);
   fann.set_activation_func_output(ActivationFunc::Sigmoid);

   let trainer = fann.on_file("train.fann");
   trainer.with_callback(1, &|nn, _, epoch| {
       eprintln!("Epoch: {}", epoch);
       eprintln!("Mean squared error: {}", nn.get_mse());

       CallbackResult::Continue
   }).train(MAX_EPOCHS, DESIRED_ERROR)?;

   fann.save("checkpoints/mnist.net")?;

   Ok(())
}
