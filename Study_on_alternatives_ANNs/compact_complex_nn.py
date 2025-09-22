"""
Compact Complex Neural Network
==============================

A streamlined implementation of a neural network operating with complex numbers.
Architecture: 3 → 2 → 1 (compact for efficiency)
Activation: Complex sigmoid function
"""

import numpy as np
import time
from typing import List, Tuple


class CompactComplexNN:
    """Compact neural network operating with complex numbers."""
    
    def __init__(self, learning_rate: float = 0.1):
        """Initialize compact complex neural network."""
        self.learning_rate = learning_rate
        
        # Compact architecture: 3 → 2 → 1
        # Initialize complex weights and biases
        self.W1 = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)) * 0.5
        self.b1 = (np.random.randn(2, 1) + 1j * np.random.randn(2, 1)) * 0.1
        self.W2 = (np.random.randn(1, 2) + 1j * np.random.randn(1, 2)) * 0.5
        self.b2 = (np.random.randn(1, 1) + 1j * np.random.randn(1, 1)) * 0.1
        
        self.training_history = []
    
    def complex_sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Complex sigmoid activation function."""
        real_part = np.real(z)
        imag_part = np.imag(z)
        
        # Clip to prevent overflow
        real_part = np.clip(real_part, -10, 10)
        imag_part = np.clip(imag_part, -10, 10)
        
        sigmoid_real = 1 / (1 + np.exp(-real_part))
        result_real = sigmoid_real * np.cos(imag_part)
        result_imag = sigmoid_real * np.sin(imag_part)
        
        return result_real + 1j * result_imag
    
    def complex_sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of complex sigmoid."""
        sig = self.complex_sigmoid(z)
        return sig * (1 - np.abs(sig)**2)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward propagation."""
        z1 = np.dot(self.W1, X) + self.b1
        a1 = self.complex_sigmoid(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.complex_sigmoid(z2)
        return a1, z1, a2
    
    def backward(self, X: np.ndarray, y: np.ndarray, a1: np.ndarray, z1: np.ndarray, a2: np.ndarray) -> float:
        """Backward propagation with complex gradients."""
        m = X.shape[1]
        
        # Output layer
        dz2 = a2 - y
        dW2 = np.dot(dz2, np.conj(a1.T)) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        
        # Hidden layer
        dz1 = np.dot(np.conj(self.W2.T), dz2) * self.complex_sigmoid_derivative(z1)
        dW1 = np.dot(dz1, np.conj(X.T)) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        
        # Update parameters
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
        # Calculate error
        error = np.mean(np.abs(a2 - y)**2)
        return error
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 500) -> List[float]:
        """Train the network."""
        errors = []
        
        for epoch in range(epochs):
            a1, z1, a2 = self.forward(X)
            error = self.backward(X, y, a1, z1, a2)
            errors.append(error)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Error = {error:.6f}")
        
        self.training_history = errors
        return errors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        _, _, a2 = self.forward(X)
        return a2


def generate_complex_data(n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data with complex numbers."""
    # Generate complex input data
    X = np.random.randn(3, n_samples) + 1j * np.random.randn(3, n_samples)
    
    # Simple complex function: y = (x1 + x2*i) * x3 / 3
    y = (X[0] + 1j * X[1]) * X[2] / 3
    y = y / (1 + np.abs(y))  # Normalize
    y = y.reshape(1, -1)
    
    return X, y


def test_compact_complex_nn() -> dict:
    """Test the compact complex neural network."""
    print("🔢 COMPACT COMPLEX NEURAL NETWORK TEST")
    print("=" * 50)
    
    # Generate data
    X_train, y_train = generate_complex_data(100)
    X_test, y_test = generate_complex_data(30)
    
    # Create and train network
    nn = CompactComplexNN(learning_rate=0.1)
    
    start_time = time.time()
    errors = nn.train(X_train, y_train, epochs=500)
    training_time = time.time() - start_time
    
    # Test performance
    predictions = nn.predict(X_test)
    test_error = np.mean(np.abs(predictions - y_test)**2)
    
    print(f"Training time: {training_time:.2f}s")
    print(f"Final training error: {errors[-1]:.6f}")
    print(f"Test error: {test_error:.6f}")
    
    return {
        'name': 'Complex',
        'architecture': '3-2-1',
        'training_time': training_time,
        'final_error': errors[-1],
        'test_error': test_error,
        'history': errors,
        'network': nn
    }


if __name__ == "__main__":
    test_compact_complex_nn()
