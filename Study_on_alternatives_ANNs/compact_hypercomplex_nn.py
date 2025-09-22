"""
Compact Hypercomplex Neural Network
===================================

A compact neural network operating with quaternions (hypercomplex numbers).
Architecture: 3 → 2 → 1 (compact for efficiency)
Activation: Quaternion sigmoid function
"""

import numpy as np
import time
import math
from typing import List, Tuple
from hypercomplex_numbers import Quaternion, QuaternionGenerator, ZERO_Q, ONE_Q, I_Q, J_Q, K_Q, quaternion_exp


class CompactHypercomplexNN:
    """Compact neural network operating with quaternions."""
    
    def __init__(self, learning_rate: float = 0.05):
        """Initialize compact hypercomplex neural network."""
        self.learning_rate = QuaternionGenerator.from_real(learning_rate)
        
        # Compact architecture: 3 → 2 → 1
        # Initialize quaternion weights and biases
        self.W1 = [[QuaternionGenerator.random_quaternion() for _ in range(3)] for _ in range(2)]
        self.b1 = [QuaternionGenerator.random_quaternion() for _ in range(2)]
        self.W2 = [[QuaternionGenerator.random_quaternion() for _ in range(2)] for _ in range(1)]
        self.b2 = [QuaternionGenerator.random_quaternion() for _ in range(1)]
        
        self.training_history = []
    
    def quaternion_sigmoid(self, q: Quaternion) -> Quaternion:
        """
        Quaternion sigmoid activation function.
        We use: σ(q) = σ(|q|) * q/|q| for non-zero quaternions
        """
        norm = q.norm()
        if norm == 0:
            return ZERO_Q
        
        # Sigmoid of the norm
        sigmoid_norm = 1.0 / (1.0 + math.exp(-min(max(norm, -10), 10)))
        
        # Scale the unit quaternion by sigmoid
        unit_q = q.scalar_multiply(1.0 / norm)
        return unit_q.scalar_multiply(sigmoid_norm)
    
    def matrix_multiply(self, weights: List[List[Quaternion]], inputs: List[Quaternion]) -> List[Quaternion]:
        """Multiply quaternion matrix by vector."""
        result = []
        for i, row in enumerate(weights):
            sum_val = ZERO_Q
            for j, w in enumerate(row):
                if j < len(inputs):
                    sum_val = sum_val + (w * inputs[j])
            result.append(sum_val)
        return result
    
    def vector_add(self, vec1: List[Quaternion], vec2: List[Quaternion]) -> List[Quaternion]:
        """Add two quaternion vectors."""
        return [a + b for a, b in zip(vec1, vec2)]
    
    def forward(self, X: List[Quaternion]) -> Tuple[List[Quaternion], List[Quaternion]]:
        """Forward propagation."""
        # Hidden layer
        z1 = self.vector_add(self.matrix_multiply(self.W1, X), self.b1)
        a1 = [self.quaternion_sigmoid(z) for z in z1]
        
        # Output layer
        z2 = self.vector_add(self.matrix_multiply(self.W2, a1), self.b2)
        a2 = [self.quaternion_sigmoid(z) for z in z2]
        
        return a1, a2
    
    def calculate_error(self, predictions: List[Quaternion], targets: List[Quaternion]) -> Quaternion:
        """Calculate mean squared error (using quaternion norm)."""
        total_error = 0.0
        for pred, target in zip(predictions, targets):
            diff = pred - target
            error = diff.norm_squared()
            total_error += error
        
        n = len(predictions)
        return QuaternionGenerator.from_real(total_error / n)
    
    def train_step(self, X: List[Quaternion], y: List[Quaternion]) -> Quaternion:
        """Single training step with simplified gradient descent."""
        a1, a2 = self.forward(X)
        
        # Calculate error
        error = self.calculate_error(a2, y)
        
        # Simplified parameter update (using approximate gradients)
        # We'll update only the real parts to maintain stability
        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                if j < len(a1):
                    # Approximate gradient using real parts
                    error_diff = a2[i] - y[i]
                    grad_real = error_diff.w * a1[j].w * self.learning_rate.w
                    
                    # Update only if gradient is reasonable
                    if abs(grad_real) < 1.0:
                        update = QuaternionGenerator.from_real(-grad_real)
                        self.W2[i][j] = self.W2[i][j] + update
        
        # Update biases
        for i in range(len(self.b2)):
            error_diff = a2[i] - y[i]
            grad_real = error_diff.w * self.learning_rate.w
            
            if abs(grad_real) < 1.0:
                update = QuaternionGenerator.from_real(-grad_real)
                self.b2[i] = self.b2[i] + update
        
        return error
    
    def train(self, X_list: List[List[Quaternion]], y_list: List[List[Quaternion]], epochs: int = 150) -> List[float]:
        """Train the network."""
        errors = []
        
        for epoch in range(epochs):
            total_error = QuaternionGenerator.from_real(0.0)
            for X, y in zip(X_list, y_list):
                error = self.train_step(X, y)
                total_error = total_error + error
            
            avg_error = total_error.scalar_multiply(1.0/len(X_list))
            error_approx = avg_error.approximate_value()
            errors.append(error_approx)
            
            if epoch % 30 == 0:
                print(f"Epoch {epoch}: Error ≈ {error_approx:.6f}")
        
        self.training_history = errors
        return errors
    
    def predict(self, X: List[Quaternion]) -> List[Quaternion]:
        """Make predictions."""
        _, a2 = self.forward(X)
        return a2


def generate_hypercomplex_data(n_samples: int = 25) -> Tuple[List[List[Quaternion]], List[List[Quaternion]]]:
    """Generate training data with quaternions."""
    X_list = []
    y_list = []
    
    basic_quaternions = QuaternionGenerator.basic_quaternions()
    
    for _ in range(n_samples):
        # Generate input
        X = [QuaternionGenerator.random_quaternion() for _ in range(3)]
        
        # Simple function: y = (q1 + q2 - q3).normalize() * 0.5
        y_val = X[0] + X[1] - X[2]
        
        # Normalize and scale to avoid extreme values
        if y_val.norm() > 0:
            y_val = y_val.normalize().scalar_multiply(0.5)
        else:
            y_val = QuaternionGenerator.from_real(0.1)
        
        X_list.append(X)
        y_list.append([y_val])
    
    return X_list, y_list


def test_compact_hypercomplex_nn() -> dict:
    """Test the compact hypercomplex neural network."""
    print("🔢 COMPACT HYPERCOMPLEX NEURAL NETWORK TEST")
    print("=" * 55)
    
    # Generate data
    X_train, y_train = generate_hypercomplex_data(40)
    X_test, y_test = generate_hypercomplex_data(12)
    
    # Create and train network
    nn = CompactHypercomplexNN(learning_rate=0.1)
    
    start_time = time.time()
    errors = nn.train(X_train, y_train, epochs=150)
    training_time = time.time() - start_time
    
    # Test performance
    test_errors = []
    for X, y_true in zip(X_test, y_test):
        predictions = nn.predict(X)
        error = nn.calculate_error(predictions, y_true)
        test_errors.append(error.approximate_value())
    
    test_error = np.mean(test_errors)
    
    print(f"Training time: {training_time:.2f}s")
    print(f"Final training error: {errors[-1]:.6f}")
    print(f"Test error: {test_error:.6f}")
    
    return {
        'name': 'Hypercomplex',
        'architecture': '3-2-1',
        'training_time': training_time,
        'final_error': errors[-1],
        'test_error': test_error,
        'history': errors,
        'network': nn
    }


if __name__ == "__main__":
    test_compact_hypercomplex_nn()
