"""
SIMPLEST EXAMPLE: Predict survival for a patient with specific values

This shows exactly what input looks like and what output you get.
"""

import numpy as np
import pandas as pd

def simple_example():
    """
    Super simple example with hardcoded patient data
    """
    
    print("=" * 70)
    print("SIMPLE SURVIVAL PREDICTION EXAMPLE")
    print("=" * 70)
    
    print("\n📋 PATIENT INPUT DATA:")
    print("-" * 70)
    
    # Example patient data (simplified)
    patient_data = {
        'age': 65,
        'gender': 'Male',
        'race': 'White',
        'tumor_stage': 'Stage IVA',
        'tumor_grade': 'G3',
        'smoking_history': 'Former',
        'alcohol_history': 'Yes',
        'hpv_status': 'Negative',
        'primary_site': 'Oral Cavity'
    }
    
    for key, value in patient_data.items():
        print(f"  {key:20s}: {value}")
    
    print("\n" + "=" * 70)
    print("🔄 PROCESSING...")
    print("=" * 70)
    print("  1. Encoding categorical variables (Male → 1, Stage IVA → 4, etc.)")
    print("  2. Combining with genetic data (mRNA, methylation, CNA)")
    print("  3. Scaling features to 0-1 range")
    print("  4. Feeding into deep learning model")
    
    # Simulated encoded features (what the model actually sees)
    encoded_features = np.array([
        65.0,   # age
        1.0,    # gender (Male=1, Female=0)
        1.0,    # race (White=1, Asian=2, etc.)
        4.0,    # tumor_stage (Stage I=1, II=2, III=3, IV=4)
        3.0,    # tumor_grade (G1=1, G2=2, G3=3)
        1.0,    # smoking (Never=0, Former=1, Current=2)
        1.0,    # alcohol (No=0, Yes=1)
        0.0,    # hpv (Negative=0, Positive=1)
        # ... plus hundreds of genetic features
    ])
    
    # Simulate model prediction (in reality, this comes from the trained neural network)
    # Higher score = higher risk
    simulated_risk_score = 2.45  # This would come from model.predict()
    
    print("\n" + "=" * 70)
    print("📊 PREDICTION OUTPUT:")
    print("=" * 70)
    
    print(f"\n  Risk Score: {simulated_risk_score:.2f}")
    print("\n  Interpretation:")
    print("    • Risk scores typically range from -2 to +4")
    print("    • Negative scores = Lower risk")
    print("    • Positive scores = Higher risk")
    print(f"    • This patient's score ({simulated_risk_score:.2f}) indicates HIGH RISK")
    
    # Risk classification
    if simulated_risk_score > 0:
        risk_category = "HIGH RISK"
        survival_estimate = "< 24 months"
        recommendation = "Aggressive treatment, close monitoring"
    else:
        risk_category = "LOW RISK"
        survival_estimate = "> 36 months"
        recommendation = "Standard treatment protocol"
    
    print(f"\n  Risk Category: {risk_category}")
    print(f"  Estimated Survival: {survival_estimate}")
    print(f"  Recommendation: {recommendation}")
    
    print("\n" + "=" * 70)
    print("📈 SURVIVAL PROBABILITY OVER TIME:")
    print("=" * 70)
    
    # Simulated survival probabilities (in reality, calculated from risk score)
    survival_probs = {
        '6 months': 0.85,
        '12 months': 0.72,
        '24 months': 0.45,
        '36 months': 0.28,
        '60 months': 0.15
    }
    
    for time, prob in survival_probs.items():
        bar = '█' * int(prob * 50)
        print(f"  {time:12s}: {bar} {prob*100:.0f}%")
    
    print("\n" + "=" * 70)
    print("✅ EXAMPLE COMPLETE!")
    print("=" * 70)
    
    print("\n💡 KEY POINTS:")
    print("  1. INPUT: Patient clinical + genetic data")
    print("  2. OUTPUT: Risk score (higher = worse prognosis)")
    print("  3. USE: Helps doctors decide treatment intensity")
    print("  4. ACCURACY: C-index ~0.75 (75% accurate in ranking patients)")

def show_multiple_patients():
    """
    Show predictions for multiple patients side-by-side
    """
    print("\n\n" + "=" * 70)
    print("COMPARING MULTIPLE PATIENTS")
    print("=" * 70)
    
    patients = [
        {
            'name': 'Patient A',
            'age': 45,
            'stage': 'Stage I',
            'smoking': 'Never',
            'risk_score': -1.2,
            'survival': '> 60 months',
            'category': 'LOW RISK'
        },
        {
            'name': 'Patient B',
            'age': 65,
            'stage': 'Stage IVA',
            'smoking': 'Former',
            'risk_score': 2.45,
            'survival': '< 24 months',
            'category': 'HIGH RISK'
        },
        {
            'name': 'Patient C',
            'age': 58,
            'stage': 'Stage II',
            'smoking': 'Current',
            'risk_score': 0.8,
            'survival': '24-36 months',
            'category': 'MODERATE RISK'
        }
    ]
    
    print("\n{:<12} {:<6} {:<12} {:<10} {:<12} {:<18} {:<15}".format(
        "Patient", "Age", "Stage", "Smoking", "Risk Score", "Est. Survival", "Category"
    ))
    print("-" * 70)
    
    for p in patients:
        print("{:<12} {:<6} {:<12} {:<10} {:<12.2f} {:<18} {:<15}".format(
            p['name'], p['age'], p['stage'], p['smoking'], 
            p['risk_score'], p['survival'], p['category']
        ))
    
    print("\n💡 Notice how:")
    print("  • Younger age + early stage + no smoking = LOW RISK")
    print("  • Older age + late stage + smoking history = HIGH RISK")

if __name__ == "__main__":
    simple_example()
    show_multiple_patients()
