import json
import streamlit as st
from typing import Dict

def get_expert_analysis(analyzer, family_analysis: dict, aero_analysis: dict) -> str:
    """Get detailed comparative analysis from LLM"""
    try:
        system_prompt = """You are an expert automotive aerodynamicist with 20+ years of experience in vehicle design and wind tunnel testing. 
        Your expertise includes:
        - Computational Fluid Dynamics (CFD) analysis
        - Wind tunnel testing and validation
        - Production vehicle aerodynamic optimization
        - Racing aerodynamics principles
        
        Guidelines for analysis:
        1. Only make statements based on established aerodynamic principles and observable features
        2. If certain aspects cannot be definitively analyzed from the given data, acknowledge the limitations
        3. Use specific technical terminology and cite fundamental aerodynamic concepts
        4. Focus on quantifiable aspects rather than subjective assessments
        5. Reference industry standard benchmarks where applicable
        
        Key areas to evaluate:
        - Pressure drag and form factor analysis
        - Boundary layer separation points
        - Wake characteristics
        - Ground effect implications
        - Reynolds number considerations
        - Surface friction effects
        
        Industry benchmarks for reference:
        - Modern sedan Cd range: 0.25-0.35
        - SUV/Minivan Cd range: 0.35-0.45
        - Sports car Cd range: 0.28-0.34
        - Hypercar Cd range: 0.20-0.30"""

        analysis_prompt = f"""Based on the provided coefficients and industry knowledge, analyze these two vehicles:

        Initial Vehicle:
        - Drag coefficient (Cd): {family_analysis['cd']:.3f}
        - Lift coefficient (Cl): {family_analysis['cl']:.3f}
        
        Aerodynamic Vehicle:
        - Drag coefficient (Cd): {aero_analysis['cd']:.3f}
        - Lift coefficient (Cl): {aero_analysis['cl']:.3f}
        
        Please provide:
        1. Technical Analysis:
           - Evaluate how each coefficient compares to industry standards
           - Identify likely flow characteristics based on the coefficients
           - Assess the relationship between drag and lift coefficients
        
        2. Performance Implications:
           - Calculate estimated power required to overcome drag at 100km/h
           - Analyze high-speed stability based on lift coefficients
           - Evaluate potential fuel efficiency impact
        
        3. Design Assessment:
           - Identify probable key design features contributing to these coefficients
           - Suggest specific areas for optimization
           - Analyze trade-offs between practicality and aerodynamic efficiency
        
        4. Recommendations:
           - Propose specific design modifications with expected improvement percentages
           - Prioritize modifications based on cost-benefit analysis
           - Consider manufacturing and practical constraints"""
        
        response = analyzer.client.invoke_model(
            modelId=analyzer.analysis_model_id,
            body=json.dumps({
                "prompt": f"\n\nHuman: {analysis_prompt}\n\nAssistant: I'll provide a detailed aerodynamic analysis based on my expertise and the given data.",
                "max_tokens_to_sample": 2000,
                "temperature": 0.3,  # Lower temperature for more focused responses
                "anthropic_version": "bedrock-2023-05-31"
            }),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        analysis = response_body.get('completion', '')
        
        return analysis
        
    except Exception as e:
        st.error(f"Error generating expert analysis: {str(e)}")
        return "Error generating comparative analysis."

def display_expert_analysis(analysis: str):
    """Format and display the expert analysis in a structured way"""
    #st.markdown("## Expert Aerodynamic Analysis")
    st.markdown("<h1 style='color: #FFA500;'>Expert Aerodynamic Analysis</h1>", unsafe_allow_html=True)
    
    # Create expandable sections for each analysis component
    with st.expander("**Technical Analysis**", expanded=True):
        st.markdown(analysis.split("2. Performance Implications")[0])
    
    with st.expander("**Performance Implications**"):
        performance = analysis.split("2. Performance Implications")[1].split("3. Design Assessment")[0]
        st.markdown(performance)
    
    with st.expander("**Design Assessment**"):
        design = analysis.split("3. Design Assessment")[1].split("4. Recommendations")[0]
        st.markdown(design)
    
    with st.expander("**Recommendations**"):
        recommendations = analysis.split("4. Recommendations")[1]
        st.markdown(recommendations)


