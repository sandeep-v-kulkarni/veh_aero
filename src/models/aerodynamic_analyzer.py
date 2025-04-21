# src/models/aerodynamic_analyzer.py

import boto3
import json
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from typing import Dict

class AerodynamicAnalyzer:
    def __init__(self):
        """Initialize the Bedrock client for both image generation and analysis"""
        self.client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.image_model_id = 'amazon.nova-canvas-v1:0'
        self.analysis_model_id = 'anthropic.claude-v2'  # Using Claude for analysis
        
    def analyze_aerodynamics(self, image: Image.Image) -> Dict[str, float]:
        """
        Analyze the aerodynamic characteristics of the vehicle using computer vision
        and LLM analysis.
        """
        # Convert PIL Image to CV2 format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Extract vehicle contours and features
        features = self._extract_vehicle_features(cv_image)
        
        # Generate analysis prompt based on features
        analysis_prompt = self._generate_analysis_prompt(features)
        
        # Get aerodynamic coefficients using Claude
        coefficients = self._get_coefficients_from_llm(analysis_prompt)
        
        return coefficients

    def _extract_vehicle_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract key aerodynamic features from the vehicle image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the main vehicle contour (largest contour)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate features
            features = {
                'frontal_area': cv2.contourArea(main_contour),
                'aspect_ratio': self._calculate_aspect_ratio(main_contour),
                'curvature': self._calculate_curvature(main_contour),
                'ground_clearance': self._estimate_ground_clearance(main_contour),
                'nose_angle': self._calculate_nose_angle(main_contour)
            }
            
            return features
        return {}

    def _calculate_aspect_ratio(self, contour: np.ndarray) -> float:
        """Calculate the aspect ratio of the vehicle"""
        x, y, w, h = cv2.boundingRect(contour)
        return w / h if h != 0 else 0

    def _calculate_curvature(self, contour: np.ndarray) -> float:
        """Calculate the average curvature of the vehicle's surface"""
        # Simplified curvature calculation
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        return hull_area / contour_area if contour_area != 0 else 1

    def _estimate_ground_clearance(self, contour: np.ndarray) -> float:
        """Estimate the ground clearance ratio"""
        x, y, w, h = cv2.boundingRect(contour)
        return y / h if h != 0 else 0

    def _calculate_nose_angle(self, contour: np.ndarray) -> float:
        """Calculate the approximate nose angle of the vehicle"""
        # Simplified angle calculation using the first few points
        if len(contour) > 10:
            front_points = contour[:10]
            vx, vy, x, y = cv2.fitLine(front_points, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx) * 180 / np.pi
            return abs(angle)
        return 0

    def _generate_analysis_prompt(self, features: Dict[str, float]) -> str:
        """Generate a prompt for the LLM based on extracted features"""
        # Convert numpy values to float scalars
        aspect_ratio = float(np.asarray(features.get('aspect_ratio', 0)).item())
        curvature = float(np.asarray(features.get('curvature', 0)).item())
        ground_clearance = float(np.asarray(features.get('ground_clearance', 0)).item())
        nose_angle = float(np.asarray(features.get('nose_angle', 0)).item())
    
        return f"""Analyze the aerodynamic characteristics of a vehicle with the following features:
            - Aspect ratio: {aspect_ratio:.2f}
            - Surface curvature ratio: {curvature:.2f}
            - Ground clearance ratio: {ground_clearance:.2f}
            - Nose angle: {nose_angle:.2f} degrees
    
            Based on these measurements and modern automotive aerodynamics principles:
            1. Estimate the drag coefficient (Cd)
            2. Estimate the lift coefficient (Cl)
            3. Provide brief justification for these values
    
            Format the response as JSON with keys: 'cd', 'cl', and 'justification'"""

    def _get_coefficients_from_llm(self, prompt: str) -> Dict[str, float]:
        """Get aerodynamic coefficients using Claude"""
        try:
            response = self.client.invoke_model(
                modelId=self.analysis_model_id,
                body=json.dumps({
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant: Let me analyze these features and provide the aerodynamic coefficients in JSON format.",
                    "max_tokens_to_sample": 1000,
                    "temperature": 0.5,
                    "anthropic_version": "bedrock-2023-05-31"
                }),
                contentType="application/json",
                accept="application/json"
            )
        except Exception as e:
            st.error(f"Error invoking model: {str(e)}")
            return {'cd': 0.30, 'cl': -0.15, 'justification': "Error invoking model"}
        
        response_body = json.loads(response['body'].read())
        # Claude's response is in the 'completion' field
        response_text = response_body.get('completion', '')
        
        # Extract JSON from the response text
        try:
            # Find JSON-like content in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
            return {
                'cd': float(analysis['cd']),
                'cl': float(analysis['cl']),
                'justification': analysis['justification']
            }
        except (json.JSONDecodeError, KeyError) as e:
            st.error(f"Error parsing response: {str(e)}")
            return {'cd': 0.30, 'cl': -0.15, 'justification': "Error parsing response"}

def get_additional_kpis(analyzer: AerodynamicAnalyzer, analysis: dict, vehicle_type: str) -> dict[str, str]:
    """Calculate additional aerodynamic KPIs using LLM"""
    try:
        prompt = f"""Based on the {vehicle_type} vehicle with:
        - Drag coefficient (Cd): {analysis['cd']}
        - Lift coefficient (Cl): {analysis['cl']}
        
        Calculate and provide the following KPIs in JSON format:
        1. Estimated top speed (mph)
        2. Fuel efficiency impact (%)
        3. High-speed stability rating (1-10)
        4. Wind noise rating (1-10)
        5. Aerodynamic efficiency ratio
        """
        
        response = analyzer.client.invoke_model(
            modelId=analyzer.analysis_model_id,
            body=json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant: Let me calculate these KPIs.",
                "max_tokens_to_sample": 1000,
                "temperature": 0.5,
                "anthropic_version": "bedrock-2023-05-31"
            }),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        response_text = response_body.get('completion', '')
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            kpis = json.loads(response_text[json_start:json_end])
            return kpis
        
    except Exception as e:
        st.error(f"Error calculating KPIs: {str(e)}")
    
    return {
        "Estimated Top Speed": "N/A",
        "Fuel Efficiency Impact": "N/A",
        "High-speed Stability": "N/A",
        "Wind Noise Rating": "N/A",
        "Aero Efficiency Ratio": "N/A"
    }
