"""
Streamlit Biochemical Neural Network Interface
=============================================

Interactive web application for controlling and visualizing the biochemical neural network.
Features real-time neurotransmitter dials and network state visualization.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Any

from biochemical_neural_system import (
    BiochemicalNeuralNetwork, NeurotransmitterConcentration, 
    NeuronState, NeurotransmitterType, create_sample_concentrations
)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'network' not in st.session_state:
        st.session_state.network = BiochemicalNeuralNetwork()
    
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    
    if 'current_concentration' not in st.session_state:
        st.session_state.current_concentration = NeurotransmitterConcentration()


def create_neurotransmitter_controls():
    """Create interactive controls for neurotransmitter concentrations."""
    st.sidebar.header("🧪 Neurotransmitter Controls")
    st.sidebar.markdown("Adjust the concentration levels of each neurotransmitter:")
    
    # Create sliders for each neurotransmitter
    concentrations = {}
    
    # Dopamine - Motivation and Reward
    concentrations['dopamine'] = st.sidebar.slider(
        "🎯 Dopamine (Motivation/Reward)",
        min_value=0.0, max_value=2.0, value=0.5, step=0.1,
        help="Controls motivation, reward processing, and motor control"
    )
    
    # Epinephrine - Arousal and Alertness
    concentrations['epinephrine'] = st.sidebar.slider(
        "⚡ Epinephrine (Arousal/Alert)",
        min_value=0.0, max_value=2.0, value=0.5, step=0.1,
        help="Controls fight-or-flight response, arousal, and attention"
    )
    
    # Serotonin - Mood and Well-being
    concentrations['serotonin'] = st.sidebar.slider(
        "😊 Serotonin (Mood/Well-being)",
        min_value=0.0, max_value=2.0, value=0.5, step=0.1,
        help="Controls mood, sleep, appetite, and overall well-being"
    )
    
    # Oxytocin - Social Bonding
    concentrations['oxytocin'] = st.sidebar.slider(
        "🤝 Oxytocin (Social Bonding)",
        min_value=0.0, max_value=2.0, value=0.5, step=0.1,
        help="Controls social bonding, trust, and empathy"
    )
    
    # Adenosine - Sleep and Fatigue
    concentrations['adenosine'] = st.sidebar.slider(
        "😴 Adenosine (Sleep/Fatigue)",
        min_value=0.0, max_value=2.0, value=0.5, step=0.1,
        help="Controls sleep pressure, fatigue, and relaxation"
    )
    
    # GABA - Inhibition and Calm
    concentrations['gaba'] = st.sidebar.slider(
        "🧘 GABA (Inhibition/Calm)",
        min_value=0.0, max_value=2.0, value=0.5, step=0.1,
        help="Controls inhibition, calmness, and anxiety reduction"
    )
    
    return NeurotransmitterConcentration(**concentrations)


def create_preset_buttons():
    """Create preset concentration buttons."""
    st.sidebar.header("🎚️ Preset Profiles")
    st.sidebar.markdown("Quick preset neurotransmitter profiles:")
    
    presets = create_sample_concentrations()
    
    cols = st.sidebar.columns(2)
    
    with cols[0]:
        if st.button("⚖️ Balanced"):
            return presets['Balanced']
        if st.button("🔥 Excited"):
            return presets['Excited']
        if st.button("😌 Relaxed"):
            return presets['Relaxed']
    
    with cols[1]:
        if st.button("👁️ Alert"):
            return presets['Alert']
        if st.button("🧘 Calm"):
            return presets['Calm']
        if st.button("😴 Tired"):
            return presets['Tired']
    
    return None


def visualize_network_state(network: BiochemicalNeuralNetwork, results: Dict[str, Any]):
    """Visualize the current network state."""
    # Create network grid visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Network State Grid', 'Activation Heatmap', 
                       'Neurotransmitter Effects', 'Network Summary'],
        specs=[[{"type": "scatter"}, {"type": "heatmap"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # 1. Network State Grid
    positions = []
    states = []
    colors = []
    
    state_color_map = {
        'stimulate': 'red',
        'relax': 'blue', 
        'calm': 'green',
        'neutralize': 'gray'
    }
    
    for neuron_detail in results['neuron_details']:
        row, col = neuron_detail['position']
        positions.append((col, 3-row))  # Flip Y for proper display
        states.append(neuron_detail['current_state'])
        colors.append(state_color_map.get(neuron_detail['current_state'], 'gray'))
    
    x_pos = [p[0] for p in positions]
    y_pos = [p[1] for p in positions]
    
    fig.add_trace(
        go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            marker=dict(size=30, color=colors),
            text=[s.upper()[:3] for s in states],
            textposition="middle center",
            name="Neuron States"
        ),
        row=1, col=1
    )
    
    # 2. Activation Heatmap
    activation_grid = np.zeros((4, 4))
    for neuron_detail in results['neuron_details']:
        row, col = neuron_detail['position']
        activation_grid[row, col] = neuron_detail['signal_strength']
    
    fig.add_trace(
        go.Heatmap(
            z=activation_grid,
            colorscale='Viridis',
            showscale=True
        ),
        row=1, col=2
    )
    
    # 3. Neurotransmitter Effects
    nt_names = ['Dopamine', 'Epinephrine', 'Serotonin', 'Oxytocin', 'Adenosine', 'GABA']
    nt_values = [
        st.session_state.current_concentration.dopamine,
        st.session_state.current_concentration.epinephrine,
        st.session_state.current_concentration.serotonin,
        st.session_state.current_concentration.oxytocin,
        st.session_state.current_concentration.adenosine,
        st.session_state.current_concentration.gaba
    ]
    
    fig.add_trace(
        go.Bar(x=nt_names, y=nt_values, name="Concentrations"),
        row=2, col=1
    )
    
    # 4. Network Summary Indicator
    network_state = results['network_state']
    state_score_map = {
        'stimulate': 100,
        'calm': 75,
        'neutralize': 50,
        'relax': 25
    }
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=state_score_map.get(network_state, 50),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Network State: {network_state.upper()}"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': state_color_map.get(network_state, 'gray')},
                'steps': [
                    {'range': [0, 25], 'color': "lightblue"},
                    {'range': [25, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="🧠 Biochemical Neural Network Real-Time Analysis",
        height=800,
        showlegend=False
    )
    
    return fig


def create_detailed_analysis(results: Dict[str, Any]):
    """Create detailed analysis of network processing."""
    st.header("📊 Detailed Network Analysis")
    
    # Network summary metrics
    summary = results['network_summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Neurons",
            summary['total_neurons'],
            help="Total number of neurons in the network"
        )
    
    with col2:
        st.metric(
            "Average Activation",
            f"{summary['average_activation']:.3f}",
            help="Average signal strength across all neurons"
        )
    
    with col3:
        st.metric(
            "Max Activation",
            f"{summary['max_activation']:.3f}",
            help="Strongest neuron signal in the network"
        )
    
    with col4:
        st.metric(
            "Network State",
            results['network_state'].upper(),
            help="Overall dominant state of the network"
        )
    
    # State distribution
    st.subheader("🎯 Neuron State Distribution")
    state_dist = summary['state_distribution']
    
    fig_dist = px.pie(
        values=list(state_dist.values()),
        names=list(state_dist.keys()),
        title="Distribution of Neuron States"
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Layer-by-layer analysis
    st.subheader("🔬 Layer-by-Layer Analysis")
    
    for layer_data in results['layer_outputs']:
        layer_num = layer_data['layer']
        states = layer_data['states']
        activations = layer_data['activations']
        
        with st.expander(f"Layer {layer_num + 1} Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Neuron States:**")
                for i, state in enumerate(states):
                    st.write(f"  Neuron {i+1}: {state.value}")
            
            with col2:
                st.write("**Activation Levels:**")
                for i, activation in enumerate(activations):
                    st.write(f"  Neuron {i+1}: {activation:.3f}")


def create_history_visualization():
    """Create visualization of processing history."""
    if not st.session_state.processing_history:
        st.info("No processing history available. Adjust neurotransmitter levels to see history.")
        return
    
    st.header("📈 Processing History")
    
    # Extract history data
    history_data = []
    for i, result in enumerate(st.session_state.processing_history):
        history_data.append({
            'step': i,
            'network_state': result['network_state'],
            'avg_activation': result['network_summary']['average_activation'],
            'max_activation': result['network_summary']['max_activation']
        })
    
    if not history_data:
        return
    
    df_history = pd.DataFrame(history_data)
    
    # Create time series plot
    fig_history = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Network State Over Time', 'Activation Levels Over Time']
    )
    
    # Network state over time
    state_numeric = df_history['network_state'].map({
        'stimulate': 3,
        'calm': 2,
        'neutralize': 1,
        'relax': 0
    })
    
    fig_history.add_trace(
        go.Scatter(
            x=df_history['step'],
            y=state_numeric,
            mode='lines+markers',
            name='Network State',
            line=dict(width=3)
        ),
        row=1, col=1
    )
    
    # Activation levels over time
    fig_history.add_trace(
        go.Scatter(
            x=df_history['step'],
            y=df_history['avg_activation'],
            mode='lines',
            name='Average Activation',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    fig_history.add_trace(
        go.Scatter(
            x=df_history['step'],
            y=df_history['max_activation'],
            mode='lines',
            name='Max Activation',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    fig_history.update_layout(height=600, title="🕐 Network Processing History")
    st.plotly_chart(fig_history, use_container_width=True)


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Biochemical Neural Network",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("🧠 Biochemical Neural Network Simulator")
    st.markdown("""
    **Interactive simulation of biological neural networks using neurotransmitter chemistry**
    
    This system models a 4×4 network of biochemical neurons that respond to different 
    neurotransmitter concentrations. Adjust the controls to see how brain chemistry 
    affects neural network behavior in real-time.
    """)
    
    # Create two main columns
    main_col, control_col = st.columns([2, 1])
    
    with main_col:
        # Neurotransmitter controls
        concentration = create_neurotransmitter_controls()
        
        # Preset buttons
        preset_concentration = create_preset_buttons()
        if preset_concentration:
            concentration = preset_concentration
        
        # Update session state
        st.session_state.current_concentration = concentration
        
        # Process button
        if st.sidebar.button("🔄 Process Network", type="primary"):
            with st.spinner("Processing biochemical neural network..."):
                results = st.session_state.network.process_network(concentration)
                st.session_state.processing_history.append(results)
                st.rerun()
    
    with control_col:
        # Display current concentrations
        st.subheader("🧪 Current Concentrations")
        conc_data = concentration.to_dict()
        
        for nt_name, value in conc_data.items():
            # Create a mini bar chart for each neurotransmitter
            progress_value = min(value / 2.0, 1.0)  # Normalize to [0,1]
            st.progress(progress_value, text=f"{nt_name.title()}: {value:.2f}")
    
    # Process network if we have valid concentrations
    if st.session_state.current_concentration:
        results = st.session_state.network.process_network(st.session_state.current_concentration)
        
        # Main visualization
        st.header("🔬 Network Visualization")
        network_fig = visualize_network_state(st.session_state.network, results)
        st.plotly_chart(network_fig, use_container_width=True)
        
        # Detailed analysis
        create_detailed_analysis(results)
        
        # History visualization
        create_history_visualization()
        
        # Raw data (expandable)
        with st.expander("🔍 Raw Network Data"):
            st.json(results['network_summary'])
    
    # Information sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About This System")
    st.sidebar.markdown("""
    **Neurotransmitter Effects:**
    - **Dopamine**: Motivation, reward
    - **Epinephrine**: Alertness, arousal  
    - **Serotonin**: Mood, well-being
    - **Oxytocin**: Social bonding
    - **Adenosine**: Sleep, fatigue
    - **GABA**: Calm, inhibition
    
    **Network States:**
    - **STIMULATE**: High arousal/motivation
    - **CALM**: Peaceful, balanced
    - **RELAX**: Low activity, restful
    - **NEUTRALIZE**: Balanced, neutral
    """)
    
    # Technical details
    with st.sidebar.expander("🔧 Technical Details"):
        st.markdown("""
        **Architecture**: 4×4 biochemical neurons
        **Components**: Soma, Chemo-receptors, Nerve
        **Processing**: Feedforward with biological rules
        **Simulation**: Real-time biochemical modeling
        """)


if __name__ == "__main__":
    main()
