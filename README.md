# 23AIBox-DeepRisk
# DeepRisk: A Deep Learning Approach for Genome-wide Assessment of Common Disease

DeepRisk is an advanced deep learning framework designed to identify genetic risk factors associated with common diseases. DeepRisk employs a biological knowledge-driven approach to model complex, nonlinear associations among single nucleotide polymorphisms (SNPs). This method allows for a more nuanced and effective identification of individuals at high risk of diseases, based on genome-wide genotype data.

## Getting Started

This section will guide you on how to get a copy of DeepRisk running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure your machine has the following software installed:

- Python 3.x
- Keras
- TensorFlow
- Other dependencies (refer to `requirements.txt`)

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/23AIBox/23AIBox-DeepRisk.git
   cd DeepRisk

2. Install requirements:
   ```bash
   pip install -r requirements.txt

## Files Descriptions

- keras_model.py: Contains the main DeepRisk model implemented using the Keras API. This script includes both the architecture of the neural network and the training procedure.

- layers.py: Defines custom neural network layers, including a partially connected layer used in the DeepRisk model.

- data.py: Provides utilities for data manipulation, including filtering, assembling, and extracting datasets required by the DeepRisk model.

- keras_model_cov.py: An extension of the keras_model.py script that incorporates covariate features (referred to as additional features in the manuscript) into the prediction model.

## How to use

To use the DeepRisk model for training and testing, you can run the following command:

    ```bash
    python keras_model.py

## Citation
  
