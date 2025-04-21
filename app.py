import streamlit as st
from src.models.vehicle_generator import VehicleImageGenerator
from src.models.aerodynamic_analyzer import AerodynamicAnalyzer, get_additional_kpis
from src.visual.flow_visualization import FlowVisualization, create_flow_visualization
from src.utils.feature_extraction import create_feature_visualization
from src.analysis.expert_analysis import get_expert_analysis,  display_expert_analysis
import cv2
import numpy as np

def main():
    st.set_page_config(page_title="Design Theme to Aerodynamics", layout="wide")
    
    # Custom CSS for the title with lighter colors and white text
    st.markdown("""
        <style>
        .title-box {
            background: linear-gradient(270deg, #3498db, #74b9ff);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin: 20px 0;
            position: relative;
            overflow: hidden;
        }
        
        .title-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: -50%;
            width: 150%;
            height: 100%;
            background: linear-gradient(
                90deg,
                rgba(255,255,255,0) 0%,
                rgba(255,255,255,0.2) 50%,
                rgba(255,255,255,0) 100%
            );
            transform: skewX(-20deg);
        }
        
        .title-text {
            color: white;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            margin: 0;
        }
        
        .subtitle-text {
            color: white;
            text-align: center;
            font-size: 1.2em;
            margin-top: 10px;
            font-family: 'Arial', sans-serif;
            opacity: 0.9;
        }

        .title-box:hover {
            background: linear-gradient(270deg, #2ecc71, #3498db);
            transition: background 0.3s ease;
        }
        </style>
        
        <div class="title-box">
            <h1 class="title-text">Design Theme to Aerodynamics</h1>
            <p class="subtitle-text">Advanced Aerodynamic Performance Evaluation System</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize generators and analyzer
    generator = VehicleImageGenerator()
    analyzer = AerodynamicAnalyzer()
    
    # Default prompts for both vehicle types
    family_prompt = "A photorealistic 3D render of a modern EV like Model 3, white background, side view, showing practical design with smooth surfaces, flush door handles, and closed front grille"
    aero_prompt = "A photorealistic 3D render of the same EV but with aerodynamic modifications: active rear spoiler, lowered suspension, white background, side view"
    
    # Create two columns for prompts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Initial Vehicle")
        family_prompt = st.text_area("Initial Vehicle Description", family_prompt, height=100)
        
    with col2:
        st.subheader("Aerodynamic Vehicle")
        aero_prompt = st.text_area("Aerodynamic Vehicle Description", aero_prompt, height=100)
    
    if st.button("Generate & Compare Vehicles"):
        # Generate and analyze both vehicles
        with st.spinner("Generating and analyzing vehicles..."):
            # Generate initial vehicle
            family_image, family_filepath = generator.generate_image(family_prompt)
            # Generate aerodynamic vehicle
            aero_image, aero_filepath = generator.generate_image(aero_prompt)
            
            if family_image and aero_image:
                # Display generated images side by side
                img_col1, img_col2 = st.columns(2)
                
                with img_col1:
                    st.image(family_image, caption="Generated Initial Vehicle", use_container_width=True)
                
                with img_col2:
                    st.image(aero_image, caption="Generated Aerodynamic Vehicle", use_container_width=True)
                
                # Extract features for both vehicles
                try:
                    # Convert PIL images to CV2 format for feature extraction
                    family_cv_image = cv2.cvtColor(np.array(family_image), cv2.COLOR_RGB2BGR)
                    aero_cv_image = cv2.cvtColor(np.array(aero_image), cv2.COLOR_RGB2BGR)
                    
                    # Extract features
                    family_features = analyzer._extract_vehicle_features(family_cv_image)
                    aero_features = analyzer._extract_vehicle_features(aero_cv_image)

                    # Analyze both vehicles
                    family_analysis = analyzer.analyze_aerodynamics(family_image)
                    aero_analysis = analyzer.analyze_aerodynamics(aero_image)
                
                    # Display comparative analysis
                    #st.subheader("Comparative Aerodynamic Analysis")
                    st.markdown("<h1 style='color: #9370DB;'>Comparative Aerodynamic Analysis</h1>", unsafe_allow_html=True)
                
                    # Create metrics comparison
                    metric_cols = st.columns(4)
                
                    with metric_cols[0]:
                        st.metric("Initial Vehicle Cd", 
                            f"{family_analysis['cd']:.3f}", 
                            f"{((aero_analysis['cd'] - family_analysis['cd'])/family_analysis['cd']*100):.1f}%")
                
                    with metric_cols[1]:
                        st.metric("Aero Vehicle Cd", 
                            f"{aero_analysis['cd']:.3f}")
                
                    with metric_cols[2]:
                        st.metric("Initial Vehicle Cl", 
                            f"{family_analysis['cl']:.3f}", 
                            f"{((aero_analysis['cl'] - family_analysis['cl'])/family_analysis['cl']*100):.1f}%")
                
                    with metric_cols[3]:
                        st.metric("Aero Vehicle Cl", 
                            f"{aero_analysis['cl']:.3f}")
                
                    # Additional KPIs
                    #st.subheader("Detailed Aerodynamic KPIs")
                    st.markdown("<h1 style='color: #9370DB;'>Detailed Aerodynamic KPIs</h1>", unsafe_allow_html=True)
                    kpi_cols = st.columns(2)
                    
                    # Extract additional KPIs using LLM
                    family_kpis = get_additional_kpis(analyzer, family_analysis, "family")
                    aero_kpis = get_additional_kpis(analyzer, aero_analysis, "aerodynamic")
                    
                    with kpi_cols[0]:
                        st.markdown("### Initial Vehicle KPIs")
                        for kpi, value in family_kpis.items():
                            st.metric(kpi, value)
                    
                    with kpi_cols[1]:
                        st.markdown("### Aerodynamic Vehicle KPIs")
                        for kpi, value in aero_kpis.items():
                            st.metric(kpi, value)
                    
                    # Comparative Analysis
                    st.subheader("Analysis Justification")
                    
                    analysis_cols = st.columns(2)
                    with analysis_cols[0]:
                        st.markdown("### Initial Vehicle Analysis")
                        st.write(family_analysis['justification'])
                    
                    with analysis_cols[1]:
                        st.markdown("### Aerodynamic Vehicle Analysis")
                        st.write(aero_analysis['justification'])
                    
                    # Feature Detection Visualization
                    st.subheader("Feature Detection Comparison")
                    viz_cols = st.columns(2)
                    
                    with viz_cols[0]:
                        family_viz = create_feature_visualization(family_image)
                        st.image(family_viz, caption="Initial Vehicle Features", use_container_width=True)
                    
                    with viz_cols[1]:
                        aero_viz = create_feature_visualization(aero_image)
                        st.image(aero_viz, caption="Aerodynamic Vehicle Features", use_container_width=True)
                    
                    # Add flow visualization
                    try:
                        create_flow_visualization(family_features, aero_features)
                    except Exception as viz_error:
                        st.error(f"Error generating flow visualization: {str(viz_error)}")
                           
                    # Expert Analysis using LLM
                    #st.subheader("Expert Comparative Analysis")
                    st.markdown("<h1 style='color: #FFA500;'>Expert Comparative Analysis</h1>", unsafe_allow_html=True)
                    expert_analysis = get_expert_analysis(analyzer, family_analysis, aero_analysis)
                    st.markdown(expert_analysis)
                    
                except Exception as e:
                    st.error(f"Error extracting features or analyzing vehicles: {str(e)}")

if __name__ == "__main__":
    main()