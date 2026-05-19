# Neural Network

A fully connected neural network built from scratch in Rust, featuring a CLI for training and an interactive GUI application for live digit/doodle classification.

## Architecture
- **`lib`**: Core mathematical engine (tensors, forward/backward passes).
- **`cli`**: Command-line interface for data loading and model training.
- **`gui`**: Interactive desktop application for model evaluation and live testing.

## Dataset Preparation

To train or evaluate the model, you need the classic MNIST dataset. For convenience, you can download the exact binary files from the official mirror.

1. Create a `data` directory in the root of the project:

```bash
mkdir -p data
cd data
```

2. Download the compressed files:

```bash
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
```

3. Extract the files (they must be raw binaries, without the `.gz` extension):

```bash
gunzip *.gz
```

*(Note: The `data/` folder is included in `.gitignore` by default to prevent committing heavy binaries).*
