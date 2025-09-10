Neural Network from Scratch (2-Layer)
A simple yet powerful implementation of a 2-layer neural network built from scratch in Python using NumPy. This project trains the model on the synthetic make_moons dataset to classify points into two classes, demonstrating core concepts like forward propagation, backpropagation, and the Adam optimizer.

📖 Overview
This repository provides an educational implementation of a 2-layer neural network without relying on high-level frameworks like TensorFlow or PyTorch. It includes:

Forward Propagation: Computes predictions through the network.
Backpropagation: Calculates gradients to update weights.
Adam Optimizer: Optimizes weights with bias correction.
Visualization: Plots the training loss curve.

The project is designed to be beginner-friendly while offering insights into the inner workings of neural networks.

📂 Project Structure
neural-network-scratch/
│
├── main.py                  # Entry point to run training
├── requirements.txt         # Required Python packages
├── README.md                # Project documentation
├── plots/                   # Directory for loss curve images
└── src/
    ├── model.py             # Neural network functions (initialize, forward, backward)
    └── train.py             # Training loop and loss visualization


⚡ Features

2-Layer Neural Network:
Input Layer: 2 neurons (for 2D dataset)
Hidden Layer: 2 neurons with ReLU activation
Output Layer: 1 neuron with Sigmoid activation


Manual Implementation:
Forward propagation
Backpropagation
Adam optimizer with bias correction


Training Visualization:
Loss curve saved as plots/loss_curve.png


Synthetic Dataset:
Uses make_moons for binary classification




🛠️ Requirements

Python 3.8+
NumPy
scikit-learn
Matplotlib

Install dependencies with:
pip install -r requirements.txt


🚀 Getting Started

Clone the Repository:
git clone https://github.com/<your-username>/neural-network-scratch.git
cd neural-network-scratch


Install Dependencies:
pip install -r requirements.txt


Run the Training Script:
python main.py


View Results:

Training logs will be printed in the terminal.
The loss curve will be saved as plots/loss_curve.png.




🧠 How It Works
1. Data Preparation

Generates a synthetic 2D dataset using make_moons.
Splits data into training and test sets.
Transposes inputs and reshapes targets for matrix operations.

2. Parameter Initialization

Weights initialized using He initialization.
Biases initialized to zeros.

3. Forward Propagation

Hidden Layer: Z1 = W1 * X + b1, A1 = ReLU(Z1)
Output Layer: Z2 = W2 * A1 + b2, y_hat = Sigmoid(Z2)

4. Backpropagation & Optimization

Computes gradients for weights and biases in both layers.
Applies Adam optimizer with bias correction for stable updates.
Updates weights and biases iteratively.

5. Loss Calculation

Uses binary cross-entropy loss.
Plots and saves the loss curve for visualization.


🖼️ Output

Terminal Logs: Step-by-step training progress.
Loss Curve: A plot of the training loss, saved as plots/loss_curve.png.

Example Loss Curve
After 10 epochs, the loss curve typically shows smooth convergence, indicating effective learning.


💡 Notes

Educational Purpose: This project is designed to help understand the mechanics of neural networks.
Extensibility: Easily modify to use real-world datasets (e.g., Breast Cancer, Iris) or add more hidden layers.
Customization: Adjust hyperparameters like learning rate or epochs in main.py for experimentation.


📚 Future Improvements

Add support for deeper networks with configurable layers.
Implement additional activation functions (e.g., tanh, Leaky ReLU).
Include evaluation metrics like accuracy or confusion matrix.


🤝 Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.


📧 Contact
For questions or suggestions, feel free to open an issue or contact the maintainer at <qhuzaifa675@gmail.com>.

