# DNA-Classification

This repository contains Python code for classifying DNA sequences using Convolutional Neural Networks (CNNs). The models provided here are designed to classify DNA sequences into specific categories using deep learning techniques. The project focuses on implementing two CNN models: `CNN10` and `CNN1DModel`, each with different architectures for DNA sequence classification.

## Authors

- Tomer Gotesdyner
- Yanai Levi

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
  - [CNN10](#cnn10)
  - [CNN1DModel](#cnn1dmodel)
- [Contributing](#contributing)

## Introduction

DNA sequence classification is an important task in bioinformatics and genetics research. This project aims to provide a solution for automated DNA sequence classification using deep learning techniques. The CNN models implemented here are capable of processing DNA sequences and making predictions about their corresponding categories.

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/YanaiLevi/DNA-Classification.git
   cd dna-classification
   ```

2. Install the required dependencies using `pip`:

   ```bash
   pip install torch numpy ...
   ```

## Usage

To use the provided CNN models for DNA classification, just follow these two simple steps:

1. Load your DNA sequence data into the `dna-classification` folder.
2. Run the `main.py` script to initiate training and testing.

## Models

### CNN10

The `CNN10` model consists of multiple convolutional blocks followed by fully connected layers for classification. It includes batch normalization and dropout layers for improved performance. The model takes into account training and validation modes and adapts batch normalization accordingly.

### CNN1DModel

The `CNN1DModel` is designed with a series of 1D convolutional layers followed by fully connected layers for classification. The model uses max pooling for downsampling and batch normalization for regularization. It also includes dropout layers to prevent overfitting.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
