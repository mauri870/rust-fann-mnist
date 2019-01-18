## RUST FANN MNIST

MNIST trained in rust using the [FANN](http://leenissen.dk/fann/wp/) library.

## Installation

```bash
yay -S fann
# or
# sudo apt install libfann-dev libfann2
git clone https://github.com/mauri870/rust-fann-mnist
cd rust-fann-mnist
```

## Usage

### Download the dataset and generate FANN files

First you need to download the mnist dataset, for that run the following commands:

```bash
mkdir -p data
NAMES=(train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte)
for n in $NAMES; do
    echo "Downloading $n..."
    wget -qO- http://yann.lecun.com/exdb/mnist/$n.gz --show-progress | gunzip -c > data/$n
done
```

Next generate the train/test and validation files for FANN:

```bash
cargo run --bin preprocess --release
```

### Train

```bash
cargo run --bin train --release
```

Since FANN rely only on cpu computations, you can expect a training time of ~40 minutes on an octa core processor.

### Test

```bash
cargo run --bin test --release
```
