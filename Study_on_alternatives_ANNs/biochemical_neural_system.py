"""
Biochemical Neural Network System
=================================

A symbolic neural network that emulates biochemical processes using neurotransmitters.
This system models biological neurons with soma, chemo-receptors, and nerve components.

BNF Grammar:
chemo-receptor := <dopamine | epinephrine | serotonin | oxytocin | adenosine | gaba>
neuron := soma, chemo-receptor, nerve
soma := < N * chemo-receptor * nerve >

Output States: stimulate, relax, neutralize, calm
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from enum import Enum
import json
from dataclasses import dataclass
import math


class NeurotransmitterType(Enum):
    """Enumeration of neurotransmitter types."""
    DOPAMINE = "dopamine"
    EPINEPHRINE = "epinephrine"  # Also known as adrenaline
    SEROTONIN = "serotonin"
    OXYTOCIN = "oxytocin"
    ADENOSINE = "adenosine"
    GABA = "gaba"


class NeuronState(Enum):
    """Possible neuron output states."""
    STIMULATE = "stimulate"
    RELAX = "relax"
    NEUTRALIZE = "neutralize"
    CALM = "calm"


@dataclass
class NeurotransmitterConcentration:
    """Represents neurotransmitter concentration levels."""
    dopamine: float = 0.0      # Motivation, reward, motor control
    epinephrine: float = 0.0   # Fight-or-flight, arousal, attention
    serotonin: float = 0.0     # Mood, sleep, appetite, well-being
    oxytocin: float = 0.0      # Social bonding, trust, empathy
    adenosine: float = 0.0     # Sleep pressure, fatigue, relaxation
    gaba: float = 0.0          # Inhibition, calm, anxiety reduction
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'dopamine': self.dopamine,
            'epinephrine': self.epinephrine,
            'serotonin': self.serotonin,
            'oxytocin': self.oxytocin,
            'adenosine': self.adenosine,
            'gaba': self.gaba
        }
    
    def normalize(self) -> 'NeurotransmitterConcentration':
        """Normalize concentrations to [0, 1] range."""
        values = [self.dopamine, self.epinephrine, self.serotonin, 
                 self.oxytocin, self.adenosine, self.gaba]
        max_val = max(values) if max(values) > 0 else 1.0
        
        return NeurotransmitterConcentration(
            dopamine=self.dopamine / max_val,
            epinephrine=self.epinephrine / max_val,
            serotonin=self.serotonin / max_val,
            oxytocin=self.oxytocin / max_val,
            adenosine=self.adenosine / max_val,
            gaba=self.gaba / max_val
        )


class ChemoReceptor:
    """
    Chemo-receptor that responds to specific neurotransmitters.
    Models the binding and response characteristics of biological receptors.
    """
    
    def __init__(self, receptor_type: NeurotransmitterType, sensitivity: float = 1.0, 
                 threshold: float = 0.1):
        """
        Initialize chemo-receptor.
        
        Args:
            receptor_type: Type of neurotransmitter this receptor responds to
            sensitivity: How strongly the receptor responds (0.0 to 2.0)
            threshold: Minimum concentration needed for activation
        """
        self.receptor_type = receptor_type
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.activation_level = 0.0
    
    def bind_neurotransmitter(self, concentration: NeurotransmitterConcentration) -> float:
        """
        Simulate neurotransmitter binding and receptor activation.
        
        Args:
            concentration: Current neurotransmitter concentrations
            
        Returns:
            Activation level (0.0 to 1.0)
        """
        # Get concentration for this receptor's neurotransmitter type
        nt_concentration = getattr(concentration, self.receptor_type.value)
        
        # Apply threshold and sensitivity
        if nt_concentration < self.threshold:
            self.activation_level = 0.0
        else:
            # Sigmoid activation with sensitivity scaling
            adjusted_concentration = (nt_concentration - self.threshold) * self.sensitivity
            self.activation_level = 1.0 / (1.0 + math.exp(-adjusted_concentration * 5))
        
        return self.activation_level
    
    def get_effect_strength(self) -> float:
        """Get the current effect strength of this receptor."""
        return self.activation_level


class Soma:
    """
    Soma (cell body) of the neuron that integrates signals from chemo-receptors.
    Implements the biological integration and decision-making process.
    """
    
    def __init__(self, receptors: List[ChemoReceptor]):
        """
        Initialize soma with chemo-receptors.
        
        Args:
            receptors: List of chemo-receptors attached to this soma
        """
        self.receptors = receptors
        self.membrane_potential = 0.0
        self.resting_potential = -70.0  # mV (biological resting potential)
        self.threshold_potential = -55.0  # mV (action potential threshold)
        self.integration_weights = self._initialize_integration_weights()
    
    def _initialize_integration_weights(self) -> Dict[NeurotransmitterType, float]:
        """Initialize integration weights for different neurotransmitters."""
        return {
            NeurotransmitterType.DOPAMINE: 1.2,      # Excitatory, strong effect
            NeurotransmitterType.EPINEPHRINE: 1.5,   # Very excitatory
            NeurotransmitterType.SEROTONIN: 0.8,     # Modulatory, mild
            NeurotransmitterType.OXYTOCIN: 0.6,      # Modulatory, social
            NeurotransmitterType.ADENOSINE: -0.8,    # Inhibitory, sleepy
            NeurotransmitterType.GABA: -1.2          # Strongly inhibitory
        }
    
    def integrate_signals(self, concentration: NeurotransmitterConcentration) -> float:
        """
        Integrate all receptor signals to determine membrane potential.
        
        Args:
            concentration: Current neurotransmitter concentrations
            
        Returns:
            Integrated signal strength
        """
        total_signal = 0.0
        
        for receptor in self.receptors:
            # Get receptor activation
            activation = receptor.bind_neurotransmitter(concentration)
            
            # Apply integration weight
            weight = self.integration_weights[receptor.receptor_type]
            contribution = activation * weight
            total_signal += contribution
        
        # Update membrane potential (simplified model)
        self.membrane_potential = self.resting_potential + (total_signal * 30.0)
        
        # Normalize to [0, 1] for output
        normalized_signal = max(0.0, min(1.0, (total_signal + 1.0) / 2.0))
        
        return normalized_signal
    
    def is_firing(self) -> bool:
        """Check if neuron should fire (membrane potential above threshold)."""
        return self.membrane_potential > self.threshold_potential


class Nerve:
    """
    Nerve component that transmits the integrated signal and determines output state.
    Models axon and synaptic transmission.
    """
    
    def __init__(self, conduction_velocity: float = 1.0):
        """
        Initialize nerve.
        
        Args:
            conduction_velocity: Speed of signal transmission
        """
        self.conduction_velocity = conduction_velocity
        self.signal_strength = 0.0
        self.output_state = NeuronState.NEUTRALIZE
    
    def transmit_signal(self, integrated_signal: float, 
                       concentration: NeurotransmitterConcentration) -> NeuronState:
        """
        Transmit signal and determine output state based on neurotransmitter profile.
        
        Args:
            integrated_signal: Signal from soma integration
            concentration: Current neurotransmitter concentrations
            
        Returns:
            Output state of the neuron
        """
        self.signal_strength = integrated_signal * self.conduction_velocity
        
        # Determine output state based on neurotransmitter dominance
        self.output_state = self._determine_output_state(concentration, integrated_signal)
        
        return self.output_state
    
    def _determine_output_state(self, concentration: NeurotransmitterConcentration, 
                               signal_strength: float) -> NeuronState:
        """
        Determine output state based on neurotransmitter concentrations and signal.
        
        Biological rules:
        - High dopamine + epinephrine → STIMULATE
        - High serotonin + oxytocin → CALM
        - High adenosine + GABA → RELAX
        - Balanced or low all → NEUTRALIZE
        """
        # Calculate neurotransmitter group strengths
        excitatory = concentration.dopamine + concentration.epinephrine
        calming = concentration.serotonin + concentration.oxytocin
        inhibitory = concentration.adenosine + concentration.gaba
        
        # Apply signal strength as a modifier
        excitatory *= signal_strength
        calming *= signal_strength
        inhibitory *= signal_strength
        
        # Determine dominant system
        max_strength = max(excitatory, calming, inhibitory)
        
        if max_strength < 0.3:  # Low overall activity
            return NeuronState.NEUTRALIZE
        elif excitatory == max_strength:
            return NeuronState.STIMULATE
        elif inhibitory == max_strength:
            return NeuronState.RELAX
        else:  # calming == max_strength
            return NeuronState.CALM


class BiochemicalNeuron:
    """
    Complete biochemical neuron with soma, chemo-receptors, and nerve.
    Implements the full BNF grammar: neuron := soma, chemo-receptor, nerve
    """
    
    def __init__(self, neuron_id: str, receptor_config: Dict[NeurotransmitterType, float] = None):
        """
        Initialize biochemical neuron.
        
        Args:
            neuron_id: Unique identifier for this neuron
            receptor_config: Configuration for receptor sensitivities
        """
        self.neuron_id = neuron_id
        self.receptors = self._create_receptors(receptor_config or {})
        self.soma = Soma(self.receptors)
        self.nerve = Nerve()
        
        # State tracking
        self.current_state = NeuronState.NEUTRALIZE
        self.activation_history = []
        self.state_history = []
    
    def _create_receptors(self, config: Dict[NeurotransmitterType, float]) -> List[ChemoReceptor]:
        """Create chemo-receptors for all neurotransmitter types."""
        receptors = []
        
        for nt_type in NeurotransmitterType:
            sensitivity = config.get(nt_type, 1.0)
            receptor = ChemoReceptor(nt_type, sensitivity=sensitivity)
            receptors.append(receptor)
        
        return receptors
    
    def process_input(self, concentration: NeurotransmitterConcentration) -> NeuronState:
        """
        Process neurotransmitter input and return output state.
        
        Args:
            concentration: Input neurotransmitter concentrations
            
        Returns:
            Output state of the neuron
        """
        # Step 1: Soma integrates receptor signals
        integrated_signal = self.soma.integrate_signals(concentration)
        
        # Step 2: Nerve transmits signal and determines output
        output_state = self.nerve.transmit_signal(integrated_signal, concentration)
        
        # Update state and history
        self.current_state = output_state
        self.activation_history.append(integrated_signal)
        self.state_history.append(output_state)
        
        # Keep history limited
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
            self.state_history.pop(0)
        
        return output_state
    
    def get_receptor_activations(self) -> Dict[str, float]:
        """Get current activation levels of all receptors."""
        return {
            receptor.receptor_type.value: receptor.activation_level
            for receptor in self.receptors
        }
    
    def get_neuron_info(self) -> Dict[str, Union[str, float, Dict]]:
        """Get comprehensive information about neuron state."""
        return {
            'neuron_id': self.neuron_id,
            'current_state': self.current_state.value,
            'membrane_potential': self.soma.membrane_potential,
            'is_firing': self.soma.is_firing(),
            'signal_strength': self.nerve.signal_strength,
            'receptor_activations': self.get_receptor_activations()
        }


class BiochemicalNeuralNetwork:
    """
    4x4 network of biochemical neurons with feedforward processing.
    Models a small neural circuit with biochemical interactions.
    """
    
    def __init__(self):
        """Initialize 4x4 biochemical neural network."""
        self.network_size = 4
        self.neurons = self._create_neuron_grid()
        self.connections = self._create_connections()
        self.network_state = NeuronState.NEUTRALIZE
        self.processing_history = []
    
    def _create_neuron_grid(self) -> List[List[BiochemicalNeuron]]:
        """Create 4x4 grid of biochemical neurons with varied characteristics."""
        neurons = []
        
        for row in range(self.network_size):
            neuron_row = []
            for col in range(self.network_size):
                # Create unique receptor configurations for diversity
                config = self._generate_receptor_config(row, col)
                neuron_id = f"N_{row}_{col}"
                neuron = BiochemicalNeuron(neuron_id, config)
                neuron_row.append(neuron)
            neurons.append(neuron_row)
        
        return neurons
    
    def _generate_receptor_config(self, row: int, col: int) -> Dict[NeurotransmitterType, float]:
        """Generate unique receptor configuration based on position."""
        # Create diversity in receptor sensitivities
        base_sensitivity = 1.0
        
        # Add position-based variations
        dopamine_sens = base_sensitivity + (row * 0.2)
        epinephrine_sens = base_sensitivity + (col * 0.2)
        serotonin_sens = base_sensitivity + ((row + col) * 0.1)
        oxytocin_sens = base_sensitivity + (abs(row - col) * 0.15)
        adenosine_sens = base_sensitivity + ((3 - row) * 0.1)
        gaba_sens = base_sensitivity + ((3 - col) * 0.1)
        
        return {
            NeurotransmitterType.DOPAMINE: dopamine_sens,
            NeurotransmitterType.EPINEPHRINE: epinephrine_sens,
            NeurotransmitterType.SEROTONIN: serotonin_sens,
            NeurotransmitterType.OXYTOCIN: oxytocin_sens,
            NeurotransmitterType.ADENOSINE: adenosine_sens,
            NeurotransmitterType.GABA: gaba_sens
        }
    
    def _create_connections(self) -> Dict[str, List[str]]:
        """Create feedforward connections between neurons."""
        connections = {}
        
        for row in range(self.network_size):
            for col in range(self.network_size):
                neuron_id = f"N_{row}_{col}"
                connected_neurons = []
                
                # Connect to neurons in next layer (feedforward)
                if row < self.network_size - 1:
                    # Connect to neurons in next row
                    for next_col in range(max(0, col-1), min(self.network_size, col+2)):
                        connected_neurons.append(f"N_{row+1}_{next_col}")
                
                connections[neuron_id] = connected_neurons
        
        return connections
    
    def process_network(self, concentration: NeurotransmitterConcentration) -> Dict[str, any]:
        """
        Process neurotransmitter input through the entire network.
        
        Args:
            concentration: Input neurotransmitter concentrations
            
        Returns:
            Network processing results
        """
        # Process each layer sequentially (feedforward)
        layer_outputs = []
        
        for row in range(self.network_size):
            layer_states = []
            layer_activations = []
            
            for col in range(self.network_size):
                neuron = self.neurons[row][col]
                
                # Process current neuron
                output_state = neuron.process_input(concentration)
                layer_states.append(output_state)
                layer_activations.append(neuron.nerve.signal_strength)
            
            layer_outputs.append({
                'layer': row,
                'states': layer_states,
                'activations': layer_activations
            })
        
        # Determine overall network state
        self.network_state = self._determine_network_state(layer_outputs)
        
        # Create comprehensive results
        results = {
            'network_state': self.network_state.value,
            'layer_outputs': layer_outputs,
            'neuron_details': self._get_all_neuron_details(),
            'network_summary': self._get_network_summary()
        }
        
        self.processing_history.append(results)
        
        # Keep history limited
        if len(self.processing_history) > 50:
            self.processing_history.pop(0)
        
        return results
    
    def _determine_network_state(self, layer_outputs: List[Dict]) -> NeuronState:
        """Determine overall network state from layer outputs."""
        # Count state occurrences across all neurons
        state_counts = {state: 0 for state in NeuronState}
        
        for layer in layer_outputs:
            for state in layer['states']:
                state_counts[state] += 1
        
        # Find dominant state
        dominant_state = max(state_counts, key=state_counts.get)
        return dominant_state
    
    def _get_all_neuron_details(self) -> List[Dict]:
        """Get detailed information for all neurons."""
        details = []
        
        for row in range(self.network_size):
            for col in range(self.network_size):
                neuron = self.neurons[row][col]
                neuron_info = neuron.get_neuron_info()
                neuron_info['position'] = (row, col)
                details.append(neuron_info)
        
        return details
    
    def _get_network_summary(self) -> Dict[str, any]:
        """Get summary statistics for the network."""
        all_states = []
        all_activations = []
        
        for row in range(self.network_size):
            for col in range(self.network_size):
                neuron = self.neurons[row][col]
                all_states.append(neuron.current_state.value)
                all_activations.append(neuron.nerve.signal_strength)
        
        return {
            'total_neurons': self.network_size * self.network_size,
            'average_activation': np.mean(all_activations),
            'max_activation': np.max(all_activations),
            'min_activation': np.min(all_activations),
            'state_distribution': {
                state.value: all_states.count(state.value) 
                for state in NeuronState
            }
        }
    
    def get_network_visualization_data(self) -> Dict[str, any]:
        """Get data formatted for network visualization."""
        nodes = []
        edges = []
        
        # Create nodes
        for row in range(self.network_size):
            for col in range(self.network_size):
                neuron = self.neurons[row][col]
                node = {
                    'id': neuron.neuron_id,
                    'position': (row, col),
                    'state': neuron.current_state.value,
                    'activation': neuron.nerve.signal_strength,
                    'membrane_potential': neuron.soma.membrane_potential,
                    'is_firing': neuron.soma.is_firing()
                }
                nodes.append(node)
        
        # Create edges based on connections
        for source_id, target_ids in self.connections.items():
            for target_id in target_ids:
                edge = {
                    'source': source_id,
                    'target': target_id,
                    'weight': 1.0  # Could be made variable
                }
                edges.append(edge)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'network_state': self.network_state.value
        }


def create_sample_concentrations() -> Dict[str, NeurotransmitterConcentration]:
    """Create sample neurotransmitter concentration profiles."""
    return {
        'Balanced': NeurotransmitterConcentration(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        'Excited': NeurotransmitterConcentration(0.9, 0.8, 0.3, 0.2, 0.1, 0.2),
        'Relaxed': NeurotransmitterConcentration(0.2, 0.1, 0.7, 0.6, 0.8, 0.9),
        'Alert': NeurotransmitterConcentration(0.7, 0.9, 0.4, 0.3, 0.2, 0.3),
        'Calm': NeurotransmitterConcentration(0.3, 0.2, 0.8, 0.9, 0.4, 0.6),
        'Tired': NeurotransmitterConcentration(0.2, 0.1, 0.4, 0.3, 0.9, 0.7)
    }


def test_biochemical_system():
    """Test the biochemical neural system."""
    print("🧠 BIOCHEMICAL NEURAL NETWORK SYSTEM TEST")
    print("=" * 60)
    
    # Create network
    network = BiochemicalNeuralNetwork()
    
    # Test with different concentration profiles
    sample_concentrations = create_sample_concentrations()
    
    print(f"Network initialized: {network.network_size}x{network.network_size} neurons")
    print(f"Total neurons: {network.network_size**2}")
    print()
    
    for profile_name, concentration in sample_concentrations.items():
        print(f"Testing profile: {profile_name}")
        print(f"Concentrations: {concentration.to_dict()}")
        
        # Process through network
        results = network.process_network(concentration)
        
        print(f"Network State: {results['network_state']}")
        print(f"Average Activation: {results['network_summary']['average_activation']:.3f}")
        print(f"State Distribution: {results['network_summary']['state_distribution']}")
        print("-" * 40)
    
    return network


if __name__ == "__main__":
    test_biochemical_system()
