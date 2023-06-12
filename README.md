# Transformer-Language-Model
This repository contains an implementation of a Transformer-based language model. The model is trained to generate text by predicting the next word in a sequence. It utilizes the Causal Self-Attention mechanism and includes positional encoding to capture the order of words in the input.

# Features
* Causal Self-Attention: The model employs self-attention mechanism to capture dependencies between words in the input sequence while ensuring causality, meaning each word can only attend to previous words.
* Transformer Block: The model consists of multiple transformer blocks, each containing a Causal Self-Attention module followed by a feed-forward neural network.
* Positional Encoding: Positional encoding is applied to the input sequence to provide information about the relative position of each word in the sequence.
* Training: The model is trained using cross-entropy loss and optimized with the Adam optimizer. Learning rate scheduling is performed using the Cosine Annealing scheduler.
* Tokenization: The model uses the AutoTokenizer from the Hugging Face transformers library to tokenize the input data.
* Dataset: The model is trained on the SST-2 dataset from the GLUE benchmark.

# Usage
* Install the required dependencies: torch, transformers, datasets, numpy, matplotlib.
* Load and preprocess the dataset using the load_dataset function from the datasets library.
* Tokenize the dataset using the AutoTokenizer from the transformers library.
* Create a DataLoader for training using the tokenized dataset and specify the batch size.
* Initialize the model by providing the vocabulary size, maximum sequence length, dimensions of keys and values (d_k), model dimension (d_model), number of attention heads (n_heads), number of transformer blocks (n_layers), and dropout probability.
* Choose the device for training (e.g., "cuda:0" for GPU or "cpu" for CPU) and move the model to the selected device.
* Define the loss function and optimizer.
* Train the model by calling the train function, providing the model, loss function, optimizer, DataLoader, and number of epochs.
* Save the trained model state for future use.

# Example
The code includes an example of training the model and generating text based on a prompt. The model is trained for a specified number of epochs, and the training progress and loss are printed. After training, the model can be used to generate text by providing a prompt string. The model will predict the next word in the sequence and continue generating text until it reaches a specified length or a predefined end token.

__Note__: Some parts of the code are commented out and may require modification to run properly, such as specifying file paths for saving the model state.

# References
* [__Attention Is All You Need__](https://arxiv.org/abs/1706.03762) by Vaswani et al.
* [__Hugging Face Transformers__](https://huggingface.co/docs) library documentation.
* [__GLUE__](https://gluebenchmark.com/) benchmark.
