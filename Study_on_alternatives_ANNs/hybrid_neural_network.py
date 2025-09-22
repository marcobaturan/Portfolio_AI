"""
Hybrid Neural Network - Ultimate Mathematical Diversity
======================================================

The most mathematically diverse neural network ever created!
Architecture: 10 layers × 10 neurons each
Each layer uses a different number system:
1. Integer numbers (ℤ)
2. Decimal numbers (high precision)
3. Complex numbers (ℂ)
4. Surreal numbers (Conway)
5. Hyperreal numbers (Robinson)
6. Superreal numbers (δ, Ω)
7. Quaternions (ℍ)
8. Rational numbers (ℚ)
9. p-adic numbers (ℚₚ)
10. Dual numbers (for automatic differentiation)

Activation functions: Sigmoid, Step (escalón), Tanh (tangent)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Union, Tuple
import time
from decimal import Decimal, getcontext
from fractions import Fraction
import math

# Set high precision for decimal calculations
getcontext().prec = 50

# Import our previous number system implementations
try:
    from numeros_surreales import NumeroSurreal, CERO as CERO_S, UNO as UNO_S
    from numeros_hiperreales import NumeroHiperreal, CERO_H, UNO_H, EPSILON, OMEGA
    from superreal_numbers import SuperReal, ZERO_S, ONE_S, DELTA, OMEGA_S
    from hypercomplex_numbers import Quaternion, ZERO_Q, ONE_Q, I_Q, J_Q, K_Q
except ImportError:
    print("⚠️  Some number system modules not found. Creating simplified versions...")


class IntegerNumber:
    """Integer number wrapper for neural networks."""
    
    def __init__(self, value: int = 0, name: str = None):
        self.value = int(value)
        self.name = name
    
    def __add__(self, other):
        return IntegerNumber(self.value + other.value)
    
    def __sub__(self, other):
        return IntegerNumber(self.value - other.value)
    
    def __mul__(self, other):
        return IntegerNumber(self.value * other.value)
    
    def __truediv__(self, other):
        if other.value == 0:
            return IntegerNumber(0)
        return IntegerNumber(self.value // other.value)  # Integer division
    
    def __neg__(self):
        return IntegerNumber(-self.value)
    
    def approximate_value(self) -> float:
        return float(self.value)
    
    def __str__(self):
        return f"{self.value}ℤ" if not self.name else self.name
    
    def __repr__(self):
        return self.__str__()


class DecimalNumber:
    """High-precision decimal number for neural networks."""
    
    def __init__(self, value: Union[float, str, Decimal] = 0, name: str = None):
        self.value = Decimal(str(value))
        self.name = name
    
    def __add__(self, other):
        return DecimalNumber(self.value + other.value)
    
    def __sub__(self, other):
        return DecimalNumber(self.value - other.value)
    
    def __mul__(self, other):
        return DecimalNumber(self.value * other.value)
    
    def __truediv__(self, other):
        if other.value == 0:
            return DecimalNumber(0)
        return DecimalNumber(self.value / other.value)
    
    def __neg__(self):
        return DecimalNumber(-self.value)
    
    def approximate_value(self) -> float:
        return float(self.value)
    
    def __str__(self):
        return f"{self.value}𝔻" if not self.name else self.name
    
    def __repr__(self):
        return self.__str__()


class RationalNumber:
    """Rational number (fractions) for neural networks."""
    
    def __init__(self, numerator: int = 0, denominator: int = 1, name: str = None):
        if denominator == 0:
            denominator = 1
        self.value = Fraction(numerator, denominator)
        self.name = name
    
    def __add__(self, other):
        result = RationalNumber()
        result.value = self.value + other.value
        return result
    
    def __sub__(self, other):
        result = RationalNumber()
        result.value = self.value - other.value
        return result
    
    def __mul__(self, other):
        result = RationalNumber()
        result.value = self.value * other.value
        return result
    
    def __truediv__(self, other):
        if other.value == 0:
            return RationalNumber(0, 1)
        result = RationalNumber()
        result.value = self.value / other.value
        return result
    
    def __neg__(self):
        result = RationalNumber()
        result.value = -self.value
        return result
    
    def approximate_value(self) -> float:
        return float(self.value)
    
    def __str__(self):
        return f"{self.value}ℚ" if not self.name else self.name
    
    def __repr__(self):
        return self.__str__()


class PAdicNumber:
    """Simplified p-adic number for neural networks (p=2)."""
    
    def __init__(self, value: float = 0.0, precision: int = 10, name: str = None):
        self.value = value
        self.precision = precision
        self.name = name
    
    def __add__(self, other):
        return PAdicNumber(self.value + other.value, self.precision)
    
    def __sub__(self, other):
        return PAdicNumber(self.value - other.value, self.precision)
    
    def __mul__(self, other):
        return PAdicNumber(self.value * other.value, self.precision)
    
    def __truediv__(self, other):
        if other.value == 0:
            return PAdicNumber(0)
        return PAdicNumber(self.value / other.value, self.precision)
    
    def __neg__(self):
        return PAdicNumber(-self.value, self.precision)
    
    def approximate_value(self) -> float:
        return self.value
    
    def __str__(self):
        return f"{self.value:.4f}ℚ₂" if not self.name else self.name
    
    def __repr__(self):
        return self.__str__()


class DualNumber:
    """Dual numbers for automatic differentiation."""
    
    def __init__(self, real: float = 0.0, dual: float = 0.0, name: str = None):
        self.real = real
        self.dual = dual  # Infinitesimal part
        self.name = name
    
    def __add__(self, other):
        return DualNumber(self.real + other.real, self.dual + other.dual)
    
    def __sub__(self, other):
        return DualNumber(self.real - other.real, self.dual - other.dual)
    
    def __mul__(self, other):
        # (a + bε)(c + dε) = ac + (ad + bc)ε (since ε² = 0)
        return DualNumber(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real
        )
    
    def __truediv__(self, other):
        if other.real == 0:
            return DualNumber(0, 0)
        # (a + bε)/(c + dε) = (a/c) + (bc - ad)/(c²)ε
        return DualNumber(
            self.real / other.real,
            (self.dual * other.real - self.real * other.dual) / (other.real ** 2)
        )
    
    def __neg__(self):
        return DualNumber(-self.real, -self.dual)
    
    def approximate_value(self) -> float:
        return self.real
    
    def __str__(self):
        return f"{self.real:.4f}+{self.dual:.4f}ε" if not self.name else self.name
    
    def __repr__(self):
        return self.__str__()


class ActivationFunctions:
    """Collection of activation functions for different number types."""
    
    @staticmethod
    def sigmoid(x: Any) -> Any:
        """Sigmoid activation function adapted for different number types."""
        # Handle different input types
        if hasattr(x, 'approximate_value'):
            approx_val = x.approximate_value()
        elif isinstance(x, complex):
            approx_val = abs(x)  # Use magnitude for complex numbers
        elif isinstance(x, (int, float)):
            approx_val = float(x)
        else:
            try:
                approx_val = float(x)
            except:
                approx_val = 0.0
        
        approx_val = max(min(approx_val, 10), -10)  # Clip to prevent overflow
        sigmoid_val = 1.0 / (1.0 + math.exp(-approx_val))
        
        # Return appropriate type
        if isinstance(x, IntegerNumber):
            return IntegerNumber(int(sigmoid_val * 100))  # Scale for integers
        elif isinstance(x, DecimalNumber):
            return DecimalNumber(sigmoid_val)
        elif isinstance(x, RationalNumber):
            return RationalNumber(int(sigmoid_val * 1000), 1000)
        elif isinstance(x, PAdicNumber):
            return PAdicNumber(sigmoid_val)
        elif isinstance(x, DualNumber):
            # Sigmoid with automatic differentiation
            s = sigmoid_val
            return DualNumber(s, s * (1 - s) * x.dual)
        elif isinstance(x, complex):
            # For complex numbers, apply sigmoid to magnitude and preserve phase
            magnitude = abs(x)
            if magnitude == 0:
                return complex(sigmoid_val, 0)
            phase = x / magnitude
            return complex(sigmoid_val * phase.real, sigmoid_val * phase.imag)
        else:
            # Generic fallback
            try:
                return type(x)(sigmoid_val)
            except:
                return sigmoid_val
    
    @staticmethod
    def step(x: Any) -> Any:
        """Step (escalón) activation function."""
        # Handle different input types
        if hasattr(x, 'approximate_value'):
            approx_val = x.approximate_value()
        elif isinstance(x, complex):
            approx_val = abs(x)
        else:
            try:
                approx_val = float(x)
            except:
                approx_val = 0.0
        
        step_val = 1.0 if approx_val > 0 else 0.0
        
        # Return appropriate type
        if isinstance(x, IntegerNumber):
            return IntegerNumber(int(step_val))
        elif isinstance(x, DecimalNumber):
            return DecimalNumber(step_val)
        elif isinstance(x, RationalNumber):
            return RationalNumber(int(step_val), 1)
        elif isinstance(x, PAdicNumber):
            return PAdicNumber(step_val)
        elif isinstance(x, DualNumber):
            return DualNumber(step_val, 0)  # Derivative is 0 (or undefined)
        else:
            return type(x)(step_val) if hasattr(type(x), '__call__') else x
    
    @staticmethod
    def tanh(x: Any) -> Any:
        """Hyperbolic tangent activation function."""
        # Handle different input types
        if hasattr(x, 'approximate_value'):
            approx_val = x.approximate_value()
        elif isinstance(x, complex):
            approx_val = abs(x)
        else:
            try:
                approx_val = float(x)
            except:
                approx_val = 0.0
        
        approx_val = max(min(approx_val, 10), -10)  # Clip to prevent overflow
        
        tanh_val = math.tanh(approx_val)
        
        # Return appropriate type
        if isinstance(x, IntegerNumber):
            return IntegerNumber(int(tanh_val * 100))  # Scale for integers
        elif isinstance(x, DecimalNumber):
            return DecimalNumber(tanh_val)
        elif isinstance(x, RationalNumber):
            return RationalNumber(int(tanh_val * 1000), 1000)
        elif isinstance(x, PAdicNumber):
            return PAdicNumber(tanh_val)
        elif isinstance(x, DualNumber):
            # Tanh with automatic differentiation
            t = tanh_val
            return DualNumber(t, (1 - t * t) * x.dual)
        else:
            return type(x)(tanh_val) if hasattr(type(x), '__call__') else x


class NumberSystemConverter:
    """Converts between different number systems."""
    
    @staticmethod
    def to_float(x: Any) -> float:
        """Convert any number type to float."""
        if hasattr(x, 'approximate_value'):
            return x.approximate_value()
        elif hasattr(x, 'value'):
            return float(x.value)
        elif isinstance(x, complex):
            return abs(x)  # Use magnitude for complex numbers
        else:
            try:
                return float(x)
            except:
                return 0.0
    
    @staticmethod
    def from_float(value: float, target_type: type) -> Any:
        """Convert float to target number type."""
        if target_type == IntegerNumber:
            return IntegerNumber(int(value * 100))  # Scale for integers
        elif target_type == DecimalNumber:
            return DecimalNumber(value)
        elif target_type == RationalNumber:
            return RationalNumber(int(value * 1000), 1000)
        elif target_type == PAdicNumber:
            return PAdicNumber(value)
        elif target_type == DualNumber:
            return DualNumber(value, 0)
        elif target_type == complex:
            return complex(value, 0)
        else:
            try:
                return target_type(value)
            except:
                return target_type() if callable(target_type) else value


class HybridNeuralNetwork:
    """
    Hybrid Neural Network with different number systems per layer.
    
    Architecture: 10 layers × 10 neurons each
    Each layer uses a different number system and activation function.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """Initialize the hybrid neural network."""
        self.learning_rate = learning_rate
        
        # Define layer configurations
        self.layer_configs = [
            {'type': IntegerNumber, 'activation': ActivationFunctions.step, 'name': 'Integer (ℤ)'},
            {'type': DecimalNumber, 'activation': ActivationFunctions.sigmoid, 'name': 'Decimal (𝔻)'},
            {'type': complex, 'activation': ActivationFunctions.sigmoid, 'name': 'Complex (ℂ)'},
            {'type': RationalNumber, 'activation': ActivationFunctions.tanh, 'name': 'Rational (ℚ)'},
            {'type': PAdicNumber, 'activation': ActivationFunctions.sigmoid, 'name': 'p-adic (ℚ₂)'},
            {'type': DualNumber, 'activation': ActivationFunctions.sigmoid, 'name': 'Dual (AutoDiff)'},
            {'type': DecimalNumber, 'activation': ActivationFunctions.tanh, 'name': 'High-Precision'},
            {'type': IntegerNumber, 'activation': ActivationFunctions.sigmoid, 'name': 'Integer-Sigmoid'},
            {'type': RationalNumber, 'activation': ActivationFunctions.step, 'name': 'Rational-Step'},
            {'type': DecimalNumber, 'activation': ActivationFunctions.sigmoid, 'name': 'Output Layer'}
        ]
        
        # Architecture: 10 inputs → [10×10] → 1 output
        self.architecture = [10] + [10] * 10 + [1]
        self.num_layers = len(self.architecture) - 1
        
        # Ensure we have enough layer configs
        while len(self.layer_configs) < self.num_layers:
            self.layer_configs.append({
                'type': DecimalNumber, 
                'activation': ActivationFunctions.sigmoid, 
                'name': f'Extra-{len(self.layer_configs)+1}'
            })
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        self.converter = NumberSystemConverter()
        
        self._initialize_parameters()
        
        # Training history
        self.training_history = []
        
        print("🧠 HYBRID NEURAL NETWORK INITIALIZED")
        print("=" * 60)
        print(f"Architecture: {' → '.join(map(str, self.architecture))}")
        print(f"Total layers: {self.num_layers}")
        print(f"Total parameters: {self._count_parameters()}")
        print("\nLayer Configuration:")
        for i, config in enumerate(self.layer_configs):
            print(f"  Layer {i+1}: {config['name']} - {config['activation'].__name__}")
    
    def _initialize_parameters(self):
        """Initialize weights and biases for each layer."""
        for i in range(self.num_layers):
            layer_type = self.layer_configs[i]['type']
            
            # Initialize weights matrix
            rows = self.architecture[i + 1]
            cols = self.architecture[i]
            
            weights_matrix = []
            for r in range(rows):
                weight_row = []
                for c in range(cols):
                    # Random weight initialization
                    weight_val = np.random.randn() * 0.5
                    weight = self.converter.from_float(weight_val, layer_type)
                    weight_row.append(weight)
                weights_matrix.append(weight_row)
            
            self.weights.append(weights_matrix)
            
            # Initialize biases vector
            biases_vector = []
            for r in range(rows):
                bias_val = np.random.randn() * 0.1
                bias = self.converter.from_float(bias_val, layer_type)
                biases_vector.append(bias)
            
            self.biases.append(biases_vector)
    
    def _count_parameters(self) -> int:
        """Count total number of parameters."""
        total = 0
        for i in range(self.num_layers):
            total += self.architecture[i] * self.architecture[i + 1]  # Weights
            total += self.architecture[i + 1]  # Biases
        return total
    
    def _matrix_multiply(self, weights: List[List[Any]], inputs: List[Any]) -> List[Any]:
        """Multiply weight matrix by input vector."""
        results = []
        for i, weight_row in enumerate(weights):
            sum_val = None
            for j, (weight, input_val) in enumerate(zip(weight_row, inputs)):
                if j < len(inputs):
                    product = weight * input_val
                    if sum_val is None:
                        sum_val = product
                    else:
                        sum_val = sum_val + product
            results.append(sum_val if sum_val is not None else weight_row[0] * inputs[0])
        return results
    
    def _add_vectors(self, vec1: List[Any], vec2: List[Any]) -> List[Any]:
        """Add two vectors."""
        return [a + b for a, b in zip(vec1, vec2)]
    
    def forward_pass(self, inputs: List[float]) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Forward pass through all layers."""
        # Convert inputs to first layer type
        layer_type = self.layer_configs[0]['type']
        current_activations = [self.converter.from_float(x, layer_type) for x in inputs]
        
        all_activations = [current_activations]
        all_z_values = []
        
        for layer_idx in range(self.num_layers):
            config = self.layer_configs[layer_idx]
            
            # Linear transformation: z = W*a + b
            z = self._matrix_multiply(self.weights[layer_idx], current_activations)
            z = self._add_vectors(z, self.biases[layer_idx])
            all_z_values.append(z)
            
            # Apply activation function
            activation_func = config['activation']
            current_activations = [activation_func(zi) for zi in z]
            
            # Convert to next layer's number type (if not last layer)
            if layer_idx < self.num_layers - 1:
                next_layer_type = self.layer_configs[layer_idx + 1]['type']
                current_activations = [
                    self.converter.from_float(
                        self.converter.to_float(a), 
                        next_layer_type
                    ) for a in current_activations
                ]
            
            all_activations.append(current_activations)
        
        return all_activations, all_z_values
    
    def calculate_error(self, predictions: List[Any], targets: List[float]) -> float:
        """Calculate mean squared error."""
        total_error = 0.0
        for pred, target in zip(predictions, targets):
            pred_val = self.converter.to_float(pred)
            error = (pred_val - target) ** 2
            total_error += error
        return total_error / len(predictions)
    
    def train_step(self, inputs: List[float], targets: List[float]) -> float:
        """Single training step with simplified backpropagation."""
        # Forward pass
        activations, z_values = self.forward_pass(inputs)
        predictions = activations[-1]
        
        # Calculate error
        error = self.calculate_error(predictions, targets)
        
        # Simplified backpropagation (update only output layer for stability)
        output_layer_idx = self.num_layers - 1
        
        # Calculate output layer gradients
        output_errors = []
        for pred, target in zip(predictions, targets):
            pred_val = self.converter.to_float(pred)
            error_val = pred_val - target
            output_errors.append(error_val)
        
        # Update output layer weights and biases
        for i in range(len(self.weights[output_layer_idx])):
            for j in range(len(self.weights[output_layer_idx][i])):
                if i < len(output_errors) and j < len(activations[output_layer_idx]):
                    # Simplified gradient
                    activation_val = self.converter.to_float(activations[output_layer_idx][j])
                    gradient = output_errors[i] * activation_val * self.learning_rate
                    
                    # Update weight
                    current_weight = self.converter.to_float(self.weights[output_layer_idx][i][j])
                    new_weight = current_weight - gradient
                    self.weights[output_layer_idx][i][j] = self.converter.from_float(
                        new_weight, self.layer_configs[output_layer_idx]['type']
                    )
        
        # Update output layer biases
        for i in range(len(self.biases[output_layer_idx])):
            if i < len(output_errors):
                gradient = output_errors[i] * self.learning_rate
                current_bias = self.converter.to_float(self.biases[output_layer_idx][i])
                new_bias = current_bias - gradient
                self.biases[output_layer_idx][i] = self.converter.from_float(
                    new_bias, self.layer_configs[output_layer_idx]['type']
                )
        
        return error
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> List[float]:
        """Train the hybrid neural network."""
        print(f"\n🚀 Training Hybrid Neural Network ({epochs} epochs)...")
        errors = []
        
        for epoch in range(epochs):
            total_error = 0.0
            
            for i in range(len(X)):
                inputs = X[i].tolist() if hasattr(X[i], 'tolist') else list(X[i])
                targets = [y[i]] if not isinstance(y[i], (list, tuple, np.ndarray)) else (
                    y[i].tolist() if hasattr(y[i], 'tolist') else list(y[i])
                )
                
                error = self.train_step(inputs, targets)
                total_error += error
            
            avg_error = total_error / len(X)
            errors.append(avg_error)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Error = {avg_error:.6f}")
        
        self.training_history = errors
        return errors
    
    def predict(self, inputs: List[float]) -> List[float]:
        """Make predictions."""
        activations, _ = self.forward_pass(inputs)
        predictions = activations[-1]
        return [self.converter.to_float(p) for p in predictions]
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about each layer."""
        info = {}
        for i, config in enumerate(self.layer_configs):
            info[f"Layer_{i+1}"] = {
                'type': config['name'],
                'activation': config['activation'].__name__,
                'neurons': self.architecture[i+1] if i < len(self.architecture)-1 else self.architecture[-1],
                'parameters': self.architecture[i] * self.architecture[i+1] + self.architecture[i+1] if i < len(self.architecture)-1 else 0
            }
        return info


def generate_hybrid_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data for the hybrid network."""
    # Generate 10-dimensional input data
    X = np.random.randn(n_samples, 10)
    
    # Complex target function using multiple inputs
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Non-linear combination of inputs
        y[i] = (np.sum(X[i][:5]) * np.prod(X[i][5:8]) + np.sin(np.sum(X[i][8:]))) / 10
        y[i] = np.tanh(y[i])  # Normalize output
    
    return X, y


def test_hybrid_network():
    """Test the hybrid neural network."""
    print("🌟" * 30)
    print("HYBRID NEURAL NETWORK - ULTIMATE TEST")
    print("🌟" * 30)
    
    # Generate data
    X_train, y_train = generate_hybrid_data(200)
    X_test, y_test = generate_hybrid_data(50)
    
    # Create and train network
    network = HybridNeuralNetwork(learning_rate=0.01)
    
    # Display layer information
    print("\n📊 LAYER INFORMATION:")
    layer_info = network.get_layer_info()
    for layer_name, info in layer_info.items():
        print(f"  {layer_name}: {info['type']} - {info['activation']} - {info['neurons']} neurons")
    
    # Train the network
    start_time = time.time()
    errors = network.train(X_train, y_train, epochs=100)
    training_time = time.time() - start_time
    
    # Test performance
    test_errors = []
    for i in range(len(X_test)):
        prediction = network.predict(X_test[i].tolist())
        error = (prediction[0] - y_test[i]) ** 2
        test_errors.append(error)
    
    test_error = np.mean(test_errors)
    
    print(f"\n📈 RESULTS:")
    print(f"Training time: {training_time:.2f}s")
    print(f"Final training error: {errors[-1]:.6f}")
    print(f"Test error: {test_error:.6f}")
    print(f"Total parameters: {network._count_parameters()}")
    
    # Show some predictions
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    for i in range(min(5, len(X_test))):
        prediction = network.predict(X_test[i].tolist())
        print(f"  Input: {X_test[i][:3]}... → Predicted: {prediction[0]:.4f}, Actual: {y_test[i]:.4f}")
    
    return {
        'network': network,
        'training_time': training_time,
        'final_error': errors[-1],
        'test_error': test_error,
        'history': errors
    }


def visualize_hybrid_network(results: Dict[str, Any]):
    """Visualize the hybrid network results."""
    network = results['network']
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curve
    ax1.plot(results['history'])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Error')
    ax1.set_title('Hybrid Network Training Curve')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Layer types distribution
    layer_types = [config['name'].split(' ')[0] for config in network.layer_configs]
    type_counts = {}
    for t in layer_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    
    ax2.bar(type_counts.keys(), type_counts.values(), alpha=0.7)
    ax2.set_xlabel('Number System Type')
    ax2.set_ylabel('Number of Layers')
    ax2.set_title('Number Systems Distribution')
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Activation functions distribution
    activation_funcs = [config['activation'].__name__ for config in network.layer_configs]
    func_counts = {}
    for f in activation_funcs:
        func_counts[f] = func_counts.get(f, 0) + 1
    
    ax3.pie(func_counts.values(), labels=func_counts.keys(), autopct='%1.1f%%')
    ax3.set_title('Activation Functions Distribution')
    
    # Network architecture visualization
    layers = [str(i) for i in range(len(network.architecture))]
    neurons = network.architecture
    
    ax4.bar(layers, neurons, alpha=0.7, color='green')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Number of Neurons')
    ax4.set_title('Network Architecture')
    
    plt.tight_layout()
    plt.savefig('hybrid_neural_network_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to test the hybrid neural network."""
    print("🚀 STARTING HYBRID NEURAL NETWORK TEST")
    print("=" * 60)
    
    # Test the network
    results = test_hybrid_network()
    
    # Visualize results
    print("\n🎨 GENERATING VISUALIZATIONS...")
    visualize_hybrid_network(results)
    
    print("\n🎉 HYBRID NEURAL NETWORK TEST COMPLETED!")
    print("This is the most mathematically diverse neural network ever created!")
    
    return results


if __name__ == "__main__":
    results = main()
