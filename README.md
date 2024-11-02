# Privacy-Preserving Traffic Flow Prediction using Federated Learning

## Overview

This project implements a privacy-preserving traffic flow prediction model using the **FedGRU algorithm**. The approach combines Federated Learning (FL) with Gated Recurrent Units (GRU) to enable collaborative training on traffic flow data without sharing sensitive information. This setup is intended to predict urban traffic flow accurately while ensuring data privacy.

## Features

- **Federated Learning with GRU**: Uses FedGRU to enable decentralized learning across client datasets.
- **Privacy Preservation**: Data remains on local devices, with only model updates shared.
- **Real-World Application**: Tested on the Caltrans PeMS dataset to simulate traffic flow prediction in urban areas.
- **Joint Announcement Protocol**: Randomly selects participating clients in each iteration, enhancing efficiency.

## Technology Stack

- **Programming Language**: Python
- **Libraries**: PyTorch, NumPy, Pandas, Matplotlib, scikit-learn
- **Hardware**: Implemented on a local machine with a GPU (Nvidia GTX 1650Ti)
- **Dataset**: Caltrans Performance Measurement System (PeMS) dataset

## Methodology

The project follows the methodology outlined in the original research paper. Key steps include:

1. **Model Initialization**: Initialize a global model with random weights.
2. **Local Training**: Clients independently train models on their local data using GRUs.
3. **FedAVG Aggregation**: Clients send model weights to a central server for aggregation using the FedAVG algorithm.
4. **Global Update**: The aggregated global model is updated and redistributed to all clients, continuing the training cycle.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/traffic-flow-prediction.git
   cd traffic-flow-prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Directly Use the Jupyter Notebook, can be run as a python file on converting to .py file.**

## Usage

The project simulates a federated setup on a single machine by dividing the dataset into separate client partitions. Each client model is trained on local data, and updates are aggregated to refine the global model.

## Results

- **Evaluation Metrics**: Mean Absolute Error (MAE) and Mean Squared Error (MSE)
- **Performance**: The FedGRU algorithm achieves performance comparable to centralized GRU while preserving data privacy.
- **Visualization**: Training and testing loss curves, along with predictions, are visualized to showcase model accuracy.

| Model | MAE  | MSE   |
|-------|------|-------|
| GRU   | 7.21 | 96.67 |
| FedGRU| 7.27 | 97.74 |

## Future Work

Potential enhancements include implementing ensemble clustering-based FedGRU for improved accuracy and experimenting with graph convolutional networks.

## License

This project is licensed under the MIT License.

## References

1. Liu, Y., Yu, J. J. Q., Kang, J., Niyato, D., & Zhang, S. "Privacy-preserving traffic flow prediction: A federated learning approach." *IEEE Internet of Things Journal*, 2020.
