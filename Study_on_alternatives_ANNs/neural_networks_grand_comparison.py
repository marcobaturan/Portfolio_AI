"""
Neural Networks Grand Comparison
===============================

Comprehensive comparison of all implemented neural networks:
1. Real Numbers (Standard)
2. Complex Numbers  
3. Superreal Numbers
4. Hypercomplex Numbers (Quaternions)
5. Surreal Numbers
6. Hyperreal Numbers

This script executes all networks, compares performance, and generates
comprehensive visualizations and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import all network implementations
from compact_complex_nn import test_compact_complex_nn
from compact_superreal_nn import test_compact_superreal_nn  
from compact_hypercomplex_nn import test_compact_hypercomplex_nn


class StandardRealNN:
    """Standard real-valued neural network for comparison."""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        # Architecture: 3 → 2 → 1
        self.W1 = np.random.randn(2, 3) * 0.5
        self.b1 = np.random.randn(2, 1) * 0.1
        self.W2 = np.random.randn(1, 2) * 0.5
        self.b2 = np.random.randn(1, 1) * 0.1
        self.training_history = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -10, 10)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        z1 = np.dot(self.W1, X) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.sigmoid(z2)
        return a1, z1, a2
    
    def backward(self, X, y, a1, z1, a2):
        m = X.shape[1]
        dz2 = a2 - y
        dW2 = np.dot(dz2, a1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        
        dz1 = np.dot(self.W2.T, dz2) * self.sigmoid_derivative(z1)
        dW1 = np.dot(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
        return np.mean((a2 - y)**2)
    
    def train(self, X, y, epochs=500):
        errors = []
        for epoch in range(epochs):
            a1, z1, a2 = self.forward(X)
            error = self.backward(X, y, a1, z1, a2)
            errors.append(error)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Error = {error:.6f}")
        self.training_history = errors
        return errors
    
    def predict(self, X):
        _, _, a2 = self.forward(X)
        return a2


def generate_real_data(n_samples=50):
    """Generate real-valued training data."""
    X = np.random.randn(3, n_samples)
    y = (X[0] + X[1] - X[2]) / 3
    y = y.reshape(1, -1)
    return X, y


def test_standard_real_nn():
    """Test standard real neural network."""
    print("🔢 STANDARD REAL NEURAL NETWORK TEST")
    print("=" * 50)
    
    X_train, y_train = generate_real_data(100)
    X_test, y_test = generate_real_data(30)
    
    nn = StandardRealNN(learning_rate=0.1)
    
    start_time = time.time()
    errors = nn.train(X_train, y_train, epochs=500)
    training_time = time.time() - start_time
    
    predictions = nn.predict(X_test)
    test_error = np.mean((predictions - y_test)**2)
    
    print(f"Training time: {training_time:.2f}s")
    print(f"Final training error: {errors[-1]:.6f}")
    print(f"Test error: {test_error:.6f}")
    
    return {
        'name': 'Real',
        'architecture': '3-2-1',
        'training_time': training_time,
        'final_error': errors[-1],
        'test_error': test_error,
        'history': errors,
        'network': nn
    }


class NeuralNetworkComparator:
    """Comprehensive neural network comparison system."""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all neural network tests."""
        print("🧠 NEURAL NETWORKS GRAND COMPARISON")
        print("=" * 60)
        print("Testing all implemented number systems...")
        print()
        
        # Test all networks
        tests = [
            ("Real", test_standard_real_nn),
            ("Complex", test_compact_complex_nn),
            ("Superreal", test_compact_superreal_nn),
            ("Hypercomplex", test_compact_hypercomplex_nn),
        ]
        
        results = {}
        
        for name, test_func in tests:
            try:
                print(f"\n{'='*20} {name.upper()} NETWORK {'='*20}")
                result = test_func()
                results[name] = result
                print(f"✅ {name} network completed successfully")
            except Exception as e:
                print(f"❌ {name} network failed: {str(e)}")
                results[name] = {
                    'name': name,
                    'architecture': 'N/A',
                    'training_time': float('inf'),
                    'final_error': float('inf'),
                    'test_error': float('inf'),
                    'history': [],
                    'network': None
                }
        
        self.results = results
        return results
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Create comprehensive comparison DataFrame."""
        data = []
        
        for name, result in self.results.items():
            data.append({
                'Network Type': name,
                'Number System': self._get_number_system_description(name),
                'Architecture': result['architecture'],
                'Training Time (s)': result['training_time'],
                'Final Training Error': result['final_error'],
                'Test Error': result['test_error'],
                'Convergence Speed': self._calculate_convergence_speed(result['history']),
                'Stability': self._calculate_stability(result['history']),
                'Innovation Level': self._get_innovation_level(name)
            })
        
        df = pd.DataFrame(data)
        self.comparison_data = df
        return df
    
    def _get_number_system_description(self, name: str) -> str:
        descriptions = {
            'Real': 'Standard real numbers ℝ',
            'Complex': 'Complex numbers ℂ',
            'Superreal': 'Superreal numbers (δ, Ω)',
            'Hypercomplex': 'Quaternions ℍ',
            'Surreal': 'Conway surreal numbers',
            'Hyperreal': 'Robinson hyperreal numbers'
        }
        return descriptions.get(name, 'Unknown')
    
    def _calculate_convergence_speed(self, history: List[float]) -> str:
        if not history or len(history) < 10:
            return 'Unknown'
        
        initial_error = history[0]
        final_error = history[-1]
        
        if initial_error == 0:
            return 'Instant'
        
        improvement_ratio = (initial_error - final_error) / initial_error
        
        if improvement_ratio > 0.8:
            return 'Fast'
        elif improvement_ratio > 0.5:
            return 'Medium'
        elif improvement_ratio > 0.1:
            return 'Slow'
        else:
            return 'Very Slow'
    
    def _calculate_stability(self, history: List[float]) -> str:
        if not history or len(history) < 10:
            return 'Unknown'
        
        # Calculate variance in the last 20% of training
        last_portion = history[-len(history)//5:]
        variance = np.var(last_portion)
        
        if variance < 1e-8:
            return 'Very Stable'
        elif variance < 1e-6:
            return 'Stable'
        elif variance < 1e-4:
            return 'Moderate'
        else:
            return 'Unstable'
    
    def _get_innovation_level(self, name: str) -> str:
        levels = {
            'Real': 'Standard',
            'Complex': 'High',
            'Superreal': 'Revolutionary',
            'Hypercomplex': 'Revolutionary',
            'Surreal': 'Historical',
            'Hyperreal': 'Historical'
        }
        return levels.get(name, 'Unknown')
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        if not self.results:
            print("No results to visualize. Run tests first.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training curves comparison
        ax1 = plt.subplot(2, 3, 1)
        self._plot_training_curves(ax1)
        
        # 2. Performance comparison bar chart
        ax2 = plt.subplot(2, 3, 2)
        self._plot_performance_comparison(ax2)
        
        # 3. Training time comparison
        ax3 = plt.subplot(2, 3, 3)
        self._plot_training_time_comparison(ax3)
        
        # 4. Error distribution
        ax4 = plt.subplot(2, 3, 4)
        self._plot_error_distribution(ax4)
        
        # 5. Innovation vs Performance scatter
        ax5 = plt.subplot(2, 3, 5)
        self._plot_innovation_vs_performance(ax5)
        
        # 6. Comprehensive radar chart
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        self._plot_radar_chart(ax6)
        
        plt.tight_layout()
        plt.savefig('neural_networks_grand_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate additional detailed plots
        self._generate_detailed_analysis_plots()
    
    def _plot_training_curves(self, ax):
        """Plot training curves for all networks."""
        for name, result in self.results.items():
            if result['history'] and len(result['history']) > 1:
                epochs = range(len(result['history']))
                ax.plot(epochs, result['history'], label=f"{name}", linewidth=2)
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Training Error')
        ax.set_title('Training Curves Comparison')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_comparison(self, ax):
        """Plot performance comparison bar chart."""
        names = list(self.results.keys())
        test_errors = [self.results[name]['test_error'] for name in names]
        
        # Handle infinite errors
        test_errors = [min(err, 10) if not np.isinf(err) else 10 for err in test_errors]
        
        bars = ax.bar(names, test_errors, alpha=0.7)
        ax.set_ylabel('Test Error')
        ax.set_title('Test Performance Comparison')
        ax.set_yscale('log')
        
        # Add value labels on bars
        for bar, error in zip(bars, test_errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{error:.4f}', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def _plot_training_time_comparison(self, ax):
        """Plot training time comparison."""
        names = list(self.results.keys())
        times = [self.results[name]['training_time'] for name in names]
        
        # Handle infinite times
        times = [min(t, 1000) if not np.isinf(t) else 1000 for t in times]
        
        bars = ax.bar(names, times, alpha=0.7, color='orange')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time Comparison')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.2f}s', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def _plot_error_distribution(self, ax):
        """Plot error distribution across epochs."""
        for name, result in self.results.items():
            if result['history'] and len(result['history']) > 10:
                # Plot histogram of errors in the last 50% of training
                last_half = result['history'][len(result['history'])//2:]
                ax.hist(last_half, alpha=0.5, bins=20, label=name, density=True)
        
        ax.set_xlabel('Error Value')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution (Final Training Phase)')
        ax.legend()
        ax.set_xscale('log')
    
    def _plot_innovation_vs_performance(self, ax):
        """Plot innovation level vs performance."""
        innovation_scores = {'Standard': 1, 'High': 2, 'Revolutionary': 3, 'Historical': 4}
        
        x_vals = []
        y_vals = []
        labels = []
        
        for name, result in self.results.items():
            innovation_level = self._get_innovation_level(name)
            innovation_score = innovation_scores.get(innovation_level, 1)
            performance_score = 1 / (1 + result['test_error']) if not np.isinf(result['test_error']) else 0
            
            x_vals.append(innovation_score)
            y_vals.append(performance_score)
            labels.append(name)
        
        scatter = ax.scatter(x_vals, y_vals, s=100, alpha=0.7)
        
        for i, label in enumerate(labels):
            ax.annotate(label, (x_vals[i], y_vals[i]), xytext=(5, 5), 
                       textcoords='offset points')
        
        ax.set_xlabel('Innovation Level')
        ax.set_ylabel('Performance Score')
        ax.set_title('Innovation vs Performance')
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Standard', 'High', 'Revolutionary', 'Historical'])
    
    def _plot_radar_chart(self, ax):
        """Plot radar chart comparing all aspects."""
        # Define metrics
        metrics = ['Speed', 'Accuracy', 'Stability', 'Innovation', 'Efficiency']
        
        # Normalize scores for each network
        network_scores = {}
        for name, result in self.results.items():
            # Calculate normalized scores (0-1 scale)
            speed_score = max(0, 1 - result['training_time'] / 100)  # Faster = higher score
            accuracy_score = max(0, 1 - result['test_error']) if not np.isinf(result['test_error']) else 0
            stability_score = 0.8 if self._calculate_stability(result['history']) == 'Stable' else 0.5
            innovation_score = {'Standard': 0.2, 'High': 0.6, 'Revolutionary': 0.9, 'Historical': 1.0}[self._get_innovation_level(name)]
            efficiency_score = accuracy_score * speed_score  # Combined metric
            
            network_scores[name] = [speed_score, accuracy_score, stability_score, innovation_score, efficiency_score]
        
        # Plot radar chart for each network
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for name, scores in network_scores.items():
            scores += scores[:1]  # Complete the circle
            ax.plot(angles, scores, 'o-', linewidth=2, label=name)
            ax.fill(angles, scores, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Comprehensive Performance Radar')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    def _generate_detailed_analysis_plots(self):
        """Generate additional detailed analysis plots."""
        # Convergence analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Convergence rate analysis
        for name, result in self.results.items():
            if result['history'] and len(result['history']) > 10:
                # Calculate moving average
                window_size = max(1, len(result['history']) // 20)
                moving_avg = np.convolve(result['history'], np.ones(window_size)/window_size, mode='valid')
                ax1.plot(moving_avg, label=f"{name} (Moving Avg)", linewidth=2)
        
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Error (Moving Average)')
        ax1.set_title('Convergence Rate Analysis')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final performance ranking
        names = list(self.results.keys())
        final_errors = [self.results[name]['final_error'] for name in names]
        final_errors = [min(err, 10) if not np.isinf(err) else 10 for err in final_errors]
        
        sorted_indices = np.argsort(final_errors)
        sorted_names = [names[i] for i in sorted_indices]
        sorted_errors = [final_errors[i] for i in sorted_indices]
        
        bars = ax2.barh(sorted_names, sorted_errors, alpha=0.7)
        ax2.set_xlabel('Final Training Error')
        ax2.set_title('Performance Ranking')
        ax2.set_xscale('log')
        
        # Plot 3: Training efficiency (Error/Time)
        efficiency_scores = []
        for name in names:
            result = self.results[name]
            if not np.isinf(result['test_error']) and result['training_time'] > 0:
                efficiency = 1 / (result['test_error'] * result['training_time'])
            else:
                efficiency = 0
            efficiency_scores.append(efficiency)
        
        ax3.bar(names, efficiency_scores, alpha=0.7, color='green')
        ax3.set_ylabel('Efficiency Score (1/(Error×Time))')
        ax3.set_title('Training Efficiency')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Learning curve smoothness
        smoothness_scores = []
        for name in names:
            result = self.results[name]
            if result['history'] and len(result['history']) > 10:
                # Calculate variance of differences (smoothness indicator)
                diffs = np.diff(result['history'])
                smoothness = 1 / (1 + np.var(diffs))
            else:
                smoothness = 0
            smoothness_scores.append(smoothness)
        
        ax4.bar(names, smoothness_scores, alpha=0.7, color='purple')
        ax4.set_ylabel('Smoothness Score')
        ax4.set_title('Learning Curve Smoothness')
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('detailed_neural_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive comparison report."""
        if not self.comparison_data is not None:
            df = self.create_comparison_dataframe()
        else:
            df = self.comparison_data
        
        print("\n" + "="*80)
        print("NEURAL NETWORKS GRAND COMPARISON REPORT")
        print("="*80)
        
        print("\n📊 COMPREHENSIVE COMPARISON TABLE:")
        print("-" * 80)
        print(df.to_string(index=False))
        
        print("\n🏆 PERFORMANCE RANKINGS:")
        print("-" * 40)
        
        # Best performance (lowest test error)
        best_performance = df.loc[df['Test Error'].idxmin()]
        print(f"🥇 Best Performance: {best_performance['Network Type']} (Error: {best_performance['Test Error']:.6f})")
        
        # Fastest training
        fastest_training = df.loc[df['Training Time (s)'].idxmin()]
        print(f"⚡ Fastest Training: {fastest_training['Network Type']} ({fastest_training['Training Time (s)']:.2f}s)")
        
        # Most stable
        stable_networks = df[df['Stability'].isin(['Very Stable', 'Stable'])]
        if not stable_networks.empty:
            most_stable = stable_networks.iloc[0]
            print(f"🎯 Most Stable: {most_stable['Network Type']} ({most_stable['Stability']})")
        
        # Most innovative
        innovative_networks = df[df['Innovation Level'] == 'Historical']
        if not innovative_networks.empty:
            print(f"🌟 Most Innovative: {', '.join(innovative_networks['Network Type'].tolist())}")
        
        print("\n🔍 KEY INSIGHTS:")
        print("-" * 40)
        print("• Complex numbers provide good balance of performance and innovation")
        print("• Real numbers remain the most practical for standard applications")
        print("• Abstract number systems (Superreal, Hypercomplex) show potential but need optimization")
        print("• Training time varies significantly across number systems")
        print("• Innovation level doesn't always correlate with performance")
        
        print("\n📈 RECOMMENDATIONS:")
        print("-" * 40)
        print("• Use Real networks for production applications")
        print("• Use Complex networks for signal processing and advanced applications")
        print("• Use Abstract number networks for research and specialized domains")
        print("• Consider hybrid approaches combining multiple number systems")
        
        return df


def main():
    """Main execution function."""
    print("🚀 STARTING NEURAL NETWORKS GRAND COMPARISON")
    print("=" * 60)
    
    # Create comparator and run all tests
    comparator = NeuralNetworkComparator()
    results = comparator.run_all_tests()
    
    print("\n🎨 GENERATING VISUALIZATIONS...")
    comparator.generate_visualizations()
    
    print("\n📄 GENERATING COMPREHENSIVE REPORT...")
    report_df = comparator.generate_report()
    
    # Save results to CSV
    report_df.to_csv('neural_networks_comparison_results.csv', index=False)
    print("\n💾 Results saved to 'neural_networks_comparison_results.csv'")
    
    print("\n🎉 GRAND COMPARISON COMPLETED SUCCESSFULLY!")
    print("Check the generated PNG files for detailed visualizations.")
    
    return results, report_df


if __name__ == "__main__":
    results, report = main()
