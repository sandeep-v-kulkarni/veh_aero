import matplotlib.pyplot as plt
import numpy as np
import io
import streamlit as st

class FlowVisualization:
    def __init__(self):
        """Initialize visualization setup using matplotlib instead of VTK for thread safety"""
        self.fig = plt.figure(figsize=(10, 8))
        
    def generate_flow_data(self, features: dict, is_aerodynamic: bool = False):
        """Generate flow field data for visualization"""
        # Create computational domain
        nx, ny = 100, 60  # Using 2D for simplicity and better performance
        x = np.linspace(-5, 15, nx)
        y = np.linspace(-3, 3, ny)
        X, Y = np.meshgrid(x, y)

        # Vehicle parameters
        length = 8
        width = 2
        height = 1.5 * (1/features.get('aspect_ratio', 1.5))
        nose_angle = features.get('nose_angle', 30)

        # Calculate flow field variables
        U = np.ones_like(X) * 30  # Base velocity
        V = np.zeros_like(Y)
        P = np.zeros_like(X)

        # Modify flow around vehicle
        for i in range(nx):
            for j in range(ny):
                dx = X[j,i]
                dy = Y[j,i]
                dist = np.sqrt(dy**2)

                if -length/2 <= dx <= length/2:
                    if dist < width:
                        P[j,i] = 0.5 * (1 - (dist/width)**2)
                        
                        if is_aerodynamic:
                            U[j,i] *= 1.3
                            V[j,i] = -0.3 * dy * np.exp(-dist)
                        else:
                            U[j,i] *= 0.8
                            V[j,i] = -0.5 * dy * np.exp(-dist)

        return X, Y, U, V, P

    def create_visualization(self, features: dict, is_aerodynamic: bool = False):
        """Create matplotlib visualization"""
        X, Y, U, V, P = self.generate_flow_data(features, is_aerodynamic)
        
        # Clear previous plot
        plt.clf()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Pressure distribution
        pressure = ax1.contourf(X, Y, P, levels=20, cmap='RdBu_r')
        ax1.set_title('Pressure Distribution')
        plt.colorbar(pressure, ax=ax1, label='Pressure')
        
        # Plot 2: Flow streamlines
        speed = np.sqrt(U**2 + V**2)
        streamplot = ax2.streamplot(X, Y, U, V, color=speed, cmap='viridis',
                                  density=2, linewidth=1, arrowsize=1)
        ax2.set_title('Flow Streamlines')
        plt.colorbar(streamplot.lines, ax=ax2, label='Velocity')
        
        # Set labels
        for ax in [ax1, ax2]:
            ax.set_xlabel('Length (m)')
            ax.set_ylabel('Height (m)')
            
        # Add vehicle outline
        vehicle_x = np.linspace(-4, 4, 100)
        if is_aerodynamic:
            vehicle_y = 0.5 * np.sin(np.pi * vehicle_x / 8)
        else:
            vehicle_y = 0.8 * np.ones_like(vehicle_x)
        
        for ax in [ax1, ax2]:
            ax.plot(vehicle_x, vehicle_y, 'k-', linewidth=2, label='Vehicle')
            ax.plot(vehicle_x, -vehicle_y, 'k-', linewidth=2)
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    

def create_flow_visualization(family_features: dict, aero_features: dict):
    """Create and display flow visualization in Streamlit"""
    st.markdown("## Vehicle Flow Analysis")
    
    viz = FlowVisualization()
    
    # Create two columns for visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div style="color: #B19CD9; font-size: 20px;">Initial Vehicle Flow Pattern</div>', 
                   unsafe_allow_html=True)
        family_viz = viz.create_visualization(family_features, False)
        st.image(family_viz)
        
    with col2:
        st.markdown('<div style="color: #4CAF50; font-size: 20px;">Aerodynamic Vehicle Flow Pattern</div>', 
                   unsafe_allow_html=True)
        aero_viz = viz.create_visualization(aero_features, True)
        st.image(aero_viz)

    # Add explanation
    st.markdown("""
    ### Understanding the Visualization
    
    #### Color Mapping:
    - 🔴 Red: High pressure regions
    - ⚪ White: Neutral pressure
    - 🔵 Blue: Low pressure regions
    
    #### Flow Features:
    1. **Streamlines**: Show the path of air particles
    2. **Pressure Contours**: Indicate pressure distribution
    3. **Wake Region**: Area behind the vehicle
    4. **Stagnation Points**: Areas of high pressure
    
    #### Key Differences:
    - Initial Vehicle: Shows more turbulent flow and larger wake region
    - Aerodynamic Vehicle: Exhibits smoother flow lines and optimized pressure distribution
    """)

    # Add interactive controls in sidebar
    st.sidebar.markdown("### Visualization Controls")
    show_pressure = st.sidebar.checkbox("Show Pressure Distribution", True)
    show_streamlines = st.sidebar.checkbox("Show Streamlines", True)   
