"""
Compact Superreal Neural Network
================================

A compact neural network operating with superreal numbers.
Architecture: 3 → 2 → 1 (compact for efficiency)
Activation: Superreal sigmoid function
"""

import numpy as np
import time
import math
from typing import List, Tuple
from superreal_numbers import SuperReal, SuperRealGenerator, ZERO_S, ONE_S, DELTA, OMEGA_S, superreal_exp


class CompactSuperrealNN:
    """Compact neural network operating with superreal numbers."""
    
    def __init__(self, learning_rate: float = 0.05):
        """Initialize compact superreal neural network."""
        self.learning_rate = SuperRealGenerator.from_real(learning_rate)
        
        # Compact architecture: 3 → 2 → 1
        # Initialize superreal weights and biases
        self.W1 = [[SuperRealGenerator.random_superreal() for _ in range(3)] for _ in range(2)]
        self.b1 = [SuperRealGenerator.random_superreal() for _ in range(2)]
        self.W2 = [[SuperRealGenerator.random_superreal() for _ in range(2)] for _ in range(1)]
        self.b2 = [SuperRealGenerator.random_superreal() for _ in range(1)]
        
        self.training_history = []
    
    def superreal_sigmoid(self, x: SuperReal) -> SuperReal:
        """Superreal sigmoid activation function."""
        # For infinite inputs
        if x.infinite > 0:
            return ONE_S
        elif x.infinite < 0:
            return ZERO_S
        
        # For finite inputs: σ(a + bδ) ≈ σ(a) + σ'(a)bδ
        real_part = x.real
        if abs(real_part) > 10:
            sigmoid_real = 1.0 if real_part > 0 else 0.0
            sigmoid_derivative = 0.0
        else:
            exp_neg = math.exp(-real_part)
            sigmoid_real = 1.0 / (1.0 + exp_neg)
            sigmoid_derivative = sigmoid_real * (1.0 - sigmoid_real)
        
        return SuperReal(sigmoid_real, sigmoid_derivative * x.infinitesimal, 0)
    
    def matrix_multiply(self, weights: List[List[SuperReal]], inputs: List[SuperReal]) -> List[SuperReal]:
        """Multiply superreal matrix by vector."""
        result = []
        for i, row in enumerate(weights):
            sum_val = ZERO_S
            for j, w in enumerate(row):
                if j < len(inputs):
                    sum_val = sum_val + (w * inputs[j])
            result.append(sum_val)
        return result
    
    def vector_add(self, vec1: List[SuperReal], vec2: List[SuperReal]) -> List[SuperReal]:
        """Add two superreal vectors."""
        return [a + b for a, b in zip(vec1, vec2)]
    
    def forward(self, X: List[SuperReal]) -> Tuple[List[SuperReal], List[SuperReal]]:
        """Forward propagation."""
        # Hidden layer
        z1 = self.vector_add(self.matrix_multiply(self.W1, X), self.b1)
        a1 = [self.superreal_sigmoid(z) for z in z1]
        
        # Output layer
        z2 = self.vector_add(self.matrix_multiply(self.W2, a1), self.b2)
        a2 = [self.superreal_sigmoid(z) for z in z2]
        
        return a1, a2
    
    def calculate_error(self, predictions: List[SuperReal], targets: List[SuperReal]) -> SuperReal:
        """Calculate mean squared error."""
        total_error = ZERO_S
        for pred, target in zip(predictions, targets):
            diff = pred - target
            error = diff * diff
            total_error = total_error + error
        
        n = len(predictions)
        return total_error * SuperRealGenerator.from_real(1.0/n)
    
    def train_step(self, X: List[SuperReal], y: List[SuperReal]) -> SuperReal:
        """Single training step with simplified gradient descent."""
        a1, a2 = self.forward(X)
        
        # Calculate error
        error = self.calculate_error(a2, y)
        
        # Simplified parameter update (using approximate gradients)
        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                if j < len(a1):
                    # Approximate gradient
                    grad_approx = (a2[i] - y[i]) * a1[j] * self.learning_rate
                    # Update only the finite part to avoid instability
                    if grad_approx.infinite == 0:
                        update = SuperReal(-grad_approx.real, -grad_approx.infinitesimal, 0)
                        self.W2[i][j] = self.W2[i][j] + update
        
        # Update biases
        for i in range(len(self.b2)):
            grad_approx = (a2[i] - y[i]) * self.learning_rate
            if grad_approx.infinite == 0:
                update = SuperReal(-grad_approx.real, -grad_approx.infinitesimal, 0)
                self.b2[i] = self.b2[i] + update
        
        return error
    
    def train(self, X_list: List[List[SuperReal]], y_list: List[List[SuperReal]], epochs: int = 200) -> List[float]:
        """Train the network."""
        errors = []
        
        for epoch in range(epochs):
            total_error = ZERO_S
            for X, y in zip(X_list, y_list):
                error = self.train_step(X, y)
                total_error = total_error + error
            
            avg_error = total_error * SuperRealGenerator.from_real(1.0/len(X_list))
            error_approx = avg_error.approximate_value()
            errors.append(error_approx)
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Error ≈ {error_approx:.6f}")
        
        self.training_history = errors
        return errors
    
    def predict(self, X: List[SuperReal]) -> List[SuperReal]:
        """Make predictions."""
        _, a2 = self.forward(X)
        return a2


def generate_superreal_data(n_samples: int = 30) -> Tuple[List[List[SuperReal]], List[List[SuperReal]]]:
    """Generate training data with superreal numbers."""
    X_list = []
    y_list = []
    
    basic_numbers = SuperRealGenerator.basic_numbers()
    
    for _ in range(n_samples):
        # Generate input
        X = [SuperRealGenerator.random_superreal() for _ in range(3)]
        
        # Simple function: y = (x1 + x2 - x3) / 3
        y_val = (X[0] + X[1] - X[2]) * SuperRealGenerator.from_real(1.0/3.0)
        
        # Normalize to avoid infinite outputs
        if y_val.infinite != 0:
            y_val = SuperReal(0.5, 0.1, 0)  # Default value for infinite cases
        
        X_list.append(X)
        y_list.append([y_val])
    
    return X_list, y_list


def test_compact_superreal_nn() -> dict:
    """Test the compact superreal neural network."""
    print("🔢 COMPACT SUPERREAL NEURAL NETWORK TEST")
    print("=" * 50)
    
    # Generate data
    X_train, y_train = generate_superreal_data(50)
    X_test, y_test = generate_superreal_data(15)
    
    # Create and train network
    nn = CompactSuperrealNN(learning_rate=0.1)
    
    start_time = time.time()
    errors = nn.train(X_train, y_train, epochs=200)
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
        'name': 'Superreal',
        'architecture': '3-2-1',
        'training_time': training_time,
        'final_error': errors[-1],
        'test_error': test_error,
        'history': errors,
        'network': nn
    }


if __name__ == "__main__":
    import math
    test_compact_superreal_nn()
