NeuralNetScratch
A simple yet powerful 2-layer neural network built from scratch in Python using NumPy. This project trains on the synthetic make_moons dataset to classify points into two classes, demonstrating core concepts like forward propagation, backpropagation, and the Adam optimizer.

📖 Overview
This repository provides an educational implementation of a 2-layer neural network without high-level frameworks like TensorFlow or PyTorch. It includes:

Forward Propagation: Computes predictions through the network.
Backpropagation: Calculates gradients to update weights.
Adam Optimizer: Optimizes weights with bias correction.
Visualization: Plots training loss curve and decision boundary.

The project is beginner-friendly and offers insights into neural network mechanics.

📂 Project Structure
NeuralNetScratch/
│
├── main.py                  # Entry point to run training
├── requirements.txt         # Required Python packages
├── README.md                # Project documentation
├── LICENSE                  # MIT License file
├── .gitignore               # Ignores generated files (e.g., plots/)
├── assets/                  # Directory for README images
│   ├── loss_curve.png       # Loss curve image
│   └── decision_boundary.png # Decision boundary image
└── src/
    ├── model.py             # Neural network functions (initialize, forward, backward)
    └── train.py             # Training loop and visualization


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
Decision boundary plot saved as plots/decision_boundary.png


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
git clone https://github.com/huzaifa12466/NeuralNetScratch.git
cd NeuralNetScratch


Install Dependencies:
pip install -r requirements.txt


Run the Training Script:
python main.py


View Results:

Training logs printed in the terminal.
Loss curve saved as plots/loss_curve.png.
Decision boundary plot saved as plots/decision_boundary.png.




🧠 How It Works

Data Preparation

Generates a synthetic 2D dataset using make_moons.
Splits data into training and test sets.
Transposes inputs and reshapes targets for matrix operations.


Parameter Initialization

Weights initialized using He initialization.
Biases initialized to zeros.


Forward Propagation

Hidden Layer: Z1 = W1 * X + b1, A1 = ReLU(Z1)
Output Layer: Z2 = W2 * A1 + b2, y_hat = Sigmoid(Z2)


Backpropagation & Optimization

Computes gradients for weights and biases in both layers.
Applies Adam optimizer with bias correction for stable updates.
Updates weights and biases iteratively.


Loss Calculation

Uses binary cross-entropy loss.
Plots and saves the loss curve and decision boundary for visualization.




🖼️ Output

Terminal Logs: Step-by-step training progress.
Loss Curve: Training loss plot, saved as plots/loss_curve.png.
Decision Boundary: Classification boundary visualization, saved as plots/decision_boundary.png.

Example Loss Curve
After 10 epochs, the loss curve shows smooth convergence, indicating effective learning.

Example Decision Boundary
Visualizes how the model separates the two classes in the make_moons dataset.


💡 Notes

Educational Purpose: Helps understand neural network mechanics.
Extensibility: Modify for real-world datasets (e.g., Breast Cancer, Iris) or add layers.
Customization: Adjust hyperparameters (learning rate, epochs) in main.py.
.gitignore: The plots/ folder is ignored to avoid committing generated images.


📚 Future Improvements

Support deeper networks with configurable layers.
Add activation functions (e.g., tanh, Leaky ReLU).
Include evaluation metrics like accuracy or confusion matrix.
Add unit tests for model and training functions.


🤝 Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.


📧 Contact
For questions or suggestions, open an issue or contact the maintainer at qhuzaifa675@gmail.com.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.