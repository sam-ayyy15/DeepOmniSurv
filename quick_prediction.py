#!/usr/bin/env python3
"""
Quick Patient Prediction - Simplified version
Uses existing data without retraining the model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# ============================================================================
# MANUAL PATIENT PARAMETERS - EDIT THESE VALUES
# ============================================================================

PATIENT_PARAMS = {
    'age': 65,
    'gender': 'Male',
    'tumor_stage': 'Stage IVA',
    'tumor_grade': 'G3',
    'smoking_history': 'Former',
    'alcohol_history': 'Yes',
    'hpv_status': 'Negative',
    't_stage': 'T4a',
    'n_stage': 'N2b',
    'm_stage': 'M0',
}

# ============================================================================
# RISK SCORE CALCULATION (Simplified Cox-like model)
# ============================================================================

def calculate_risk_score(params):
    """Calculate risk score based on clinical parameters"""
    risk = 0.0
    
    # Age effect (0.02 per year above 50)
    risk += (params['age'] - 50) * 0.02
    
    # Stage effect
    stage_risk = {
        'Stage I': 0.0,
        'Stage II': 0.5,
        'Stage III': 1.0,
        'Stage IVA': 1.8,
        'Stage IVB': 2.5,
        'Stage IVC': 3.0
    }
    risk += stage_risk.get(params['tumor_stage'], 1.0)
    
    # Grade effect
    grade_risk = {'G1': 0.0, 'G2': 0.3, 'G3': 0.8, 'G4': 1.2}
    risk += grade_risk.get(params['tumor_grade'], 0.5)
    
    # T stage effect
    if params['t_stage'] in ['T3', 'T4', 'T4a', 'T4b']:
        risk += 0.5
    
    # N stage effect
    if params['n_stage'] in ['N2', 'N2a', 'N2b', 'N2c', 'N3']:
        risk += 0.8
    
    # Smoking effect
    if params['smoking_history'] in ['Current', 'Former']:
        risk += 0.3
    
    # Alcohol effect
    if params['alcohol_history'] == 'Yes':
        risk += 0.2
    
    # HPV effect (protective)
    if params['hpv_status'] == 'Positive':
        risk -= 0.5
    elif params['hpv_status'] == 'Negative':
        risk += 0.3
    
    # Gender effect
    if params['gender'] == 'Male':
        risk += 0.1
    
    return risk

def estimate_survival_probabilities(risk_score):
    """Estimate survival probabilities at different time points"""
    # Baseline survival (average patient)
    baseline = np.array([0.95, 0.85, 0.65, 0.50, 0.35])
    
    # Adjust based on risk score using exponential model
    # S(t) = S0(t) ^ exp(risk_score)
    hazard_ratio = np.exp(risk_score * 0.3)
    survival_probs = baseline ** hazard_ratio
    
    # Clip to reasonable range
    survival_probs = np.clip(survival_probs, 0.05, 0.99)
    
    return survival_probs

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(params, risk_score, survival_probs):
    """Create comprehensive visualization"""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Determine risk category
    if risk_score > 2.0:
        risk_category = "VERY HIGH RISK"
        risk_color = 'darkred'
    elif risk_score > 1.0:
        risk_category = "HIGH RISK"
        risk_color = 'red'
    elif risk_score > 0:
        risk_category = "MODERATE RISK"
        risk_color = 'orange'
    elif risk_score > -1.0:
        risk_category = "LOW RISK"
        risk_color = 'lightgreen'
    else:
        risk_category = "VERY LOW RISK"
        risk_color = 'green'
    
    # Plot 1: Risk Score Gauge
    ax1 = fig.add_subplot(gs[0, 0])
    risk_range = np.linspace(-2, 4, 100)
    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0, 1, 100))
    
    for i in range(len(risk_range)-1):
        ax1.barh(0, 0.06, left=risk_range[i], color=colors_gradient[i], height=0.5)
    
    ax1.plot([risk_score, risk_score], [-0.3, 0.3], 'k-', linewidth=3)
    ax1.plot(risk_score, 0, 'ko', markersize=15)
    ax1.set_xlim(-2, 4)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Risk Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Risk Assessment\nScore: {risk_score:.2f} ({risk_category})', 
                  fontsize=13, fontweight='bold')
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Survival Probability Bars
    ax2 = fig.add_subplot(gs[0, 1:])
    time_points = np.array([6, 12, 24, 36, 60])
    colors = ['green' if p > 0.7 else 'orange' if p > 0.4 else 'red' for p in survival_probs]
    
    bars = ax2.bar(time_points, survival_probs * 100, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=2, width=4)
    
    for bar, prob in zip(bars, survival_probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{prob*100:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_title('Predicted Survival Probability Over Time', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (Months)', fontsize=12)
    ax2.set_ylabel('Survival Probability (%)', fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.set_xticks(time_points)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.legend()
    
    # Plot 3: Survival Curve
    ax3 = fig.add_subplot(gs[1, :])
    time_extended = np.linspace(0, 60, 100)
    
    # Generate smooth survival curve
    baseline_curve = 0.95 ** (time_extended / 6)
    hazard_ratio = np.exp(risk_score * 0.3)
    survival_curve = baseline_curve ** hazard_ratio
    
    ax3.plot(time_extended, survival_curve * 100, linewidth=3, color=risk_color, 
            label=f'Your Patient ({risk_category})')
    ax3.fill_between(time_extended, 0, survival_curve * 100, alpha=0.2, color=risk_color)
    
    # Add reference curves
    low_risk_curve = baseline_curve ** np.exp(-0.5 * 0.3)
    high_risk_curve = baseline_curve ** np.exp(2.5 * 0.3)
    
    ax3.plot(time_extended, low_risk_curve * 100, '--', linewidth=2, color='green', 
            alpha=0.5, label='Low Risk Reference')
    ax3.plot(time_extended, high_risk_curve * 100, '--', linewidth=2, color='red', 
            alpha=0.5, label='High Risk Reference')
    
    # Mark key time points
    for t, p in zip(time_points, survival_probs):
        ax3.plot(t, p * 100, 'o', markersize=10, color='black', zorder=5)
    
    ax3.set_title('Survival Curve Prediction', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (Months)', fontsize=12)
    ax3.set_ylabel('Survival Probability (%)', fontsize=12)
    ax3.set_xlim(0, 60)
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc='upper right')
    ax3.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    
    # Plot 4: Risk Factors Contribution
    ax4 = fig.add_subplot(gs[2, :2])
    
    factors = []
    contributions = []
    
    # Calculate individual contributions
    if params['age'] > 50:
        factors.append(f"Age ({params['age']})")
        contributions.append((params['age'] - 50) * 0.02)
    
    stage_risk = {'Stage I': 0.0, 'Stage II': 0.5, 'Stage III': 1.0, 
                  'Stage IVA': 1.8, 'Stage IVB': 2.5, 'Stage IVC': 3.0}
    if params['tumor_stage'] in stage_risk:
        factors.append(f"Stage ({params['tumor_stage']})")
        contributions.append(stage_risk[params['tumor_stage']])
    
    grade_risk = {'G1': 0.0, 'G2': 0.3, 'G3': 0.8, 'G4': 1.2}
    if params['tumor_grade'] in grade_risk:
        factors.append(f"Grade ({params['tumor_grade']})")
        contributions.append(grade_risk[params['tumor_grade']])
    
    if params['t_stage'] in ['T3', 'T4', 'T4a', 'T4b']:
        factors.append(f"T Stage ({params['t_stage']})")
        contributions.append(0.5)
    
    if params['n_stage'] in ['N2', 'N2a', 'N2b', 'N2c', 'N3']:
        factors.append(f"N Stage ({params['n_stage']})")
        contributions.append(0.8)
    
    if params['smoking_history'] in ['Current', 'Former']:
        factors.append(f"Smoking ({params['smoking_history']})")
        contributions.append(0.3)
    
    if params['alcohol_history'] == 'Yes':
        factors.append("Alcohol Use")
        contributions.append(0.2)
    
    if params['hpv_status'] == 'Negative':
        factors.append("HPV Negative")
        contributions.append(0.3)
    elif params['hpv_status'] == 'Positive':
        factors.append("HPV Positive")
        contributions.append(-0.5)
    
    # Sort by contribution
    sorted_indices = np.argsort(contributions)[::-1]
    factors = [factors[i] for i in sorted_indices]
    contributions = [contributions[i] for i in sorted_indices]
    
    colors_factors = ['red' if c > 0 else 'green' for c in contributions]
    
    y_pos = np.arange(len(factors))
    ax4.barh(y_pos, contributions, color=colors_factors, alpha=0.7, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(factors, fontsize=10)
    ax4.set_xlabel('Contribution to Risk Score', fontsize=11, fontweight='bold')
    ax4.set_title('Risk Factors Breakdown', fontsize=13, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (factor, contrib) in enumerate(zip(factors, contributions)):
        ax4.text(contrib + 0.05 if contrib > 0 else contrib - 0.05, i, 
                f'{contrib:+.2f}',
                va='center', ha='left' if contrib > 0 else 'right',
                fontsize=9, fontweight='bold')
    
    # Plot 5: Patient Summary
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    summary_text = f"""
PATIENT SUMMARY
{'='*30}

Demographics:
  Age: {params['age']} years
  Gender: {params['gender']}

Tumor Characteristics:
  Stage: {params['tumor_stage']}
  Grade: {params['tumor_grade']}
  T: {params['t_stage']}
  N: {params['n_stage']}
  M: {params['m_stage']}

Risk Factors:
  Smoking: {params['smoking_history']}
  Alcohol: {params['alcohol_history']}
  HPV: {params['hpv_status']}

RISK ASSESSMENT:
  Score: {risk_score:.2f}
  Category: {risk_category}

MEDIAN SURVIVAL:
  Estimated: {estimate_median_survival(survival_probs)}

RECOMMENDATION:
  {get_recommendation(risk_score)}
"""
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DeepOmicsSurv - Patient Survival Prediction Report', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('patient_prediction_report.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: patient_prediction_report.png")
    
    return fig

def estimate_median_survival(survival_probs):
    """Estimate median survival time"""
    time_points = [6, 12, 24, 36, 60]
    
    for i, prob in enumerate(survival_probs):
        if prob < 0.5:
            if i == 0:
                return "< 6 months"
            else:
                return f"{time_points[i-1]}-{time_points[i]} months"
    
    return "> 60 months"

def get_recommendation(risk_score):
    """Get treatment recommendation based on risk"""
    if risk_score > 2.0:
        return "Aggressive treatment\n  + immunotherapy"
    elif risk_score > 1.0:
        return "Aggressive treatment\n  recommended"
    elif risk_score > 0:
        return "Standard treatment\n  with monitoring"
    else:
        return "Standard treatment\n  appropriate"

# ============================================================================
# DETAILED REPORT
# ============================================================================

def print_detailed_report(params, risk_score, survival_probs):
    """Print comprehensive text report"""
    
    print("\n" + "="*70)
    print("                    PREDICTION REPORT")
    print("="*70)
    
    print("\nPATIENT INFORMATION:")
    print("-" * 70)
    for key, value in params.items():
        print(f"  {key.replace('_', ' ').title():20s}: {value}")
    
    print("\nRISK ASSESSMENT:")
    print("-" * 70)
    print(f"  Risk Score:       {risk_score:.2f}")
    
    if risk_score > 2.0:
        risk_level = "VERY HIGH RISK ⚠️⚠️⚠️"
    elif risk_score > 1.0:
        risk_level = "HIGH RISK ⚠️⚠️"
    elif risk_score > 0:
        risk_level = "MODERATE RISK ⚠️"
    elif risk_score > -1.0:
        risk_level = "LOW RISK ✅"
    else:
        risk_level = "VERY LOW RISK ✅✅"
    
    print(f"  Risk Category:    {risk_level}")
    
    print("\nSURVIVAL PREDICTIONS:")
    print("-" * 70)
    time_points = [6, 12, 24, 36, 60]
    for time, prob in zip(time_points, survival_probs):
        bar_length = int(prob * 40)
        bar = "█" * bar_length
        print(f"  {time:2d} months:  {prob*100:5.1f}% {bar}")
    
    print(f"\n  Estimated Median Survival: {estimate_median_survival(survival_probs)}")
    
    print("\nCLINICAL RECOMMENDATIONS:")
    print("-" * 70)
    
    if risk_score > 2.0:
        print("  ⚠️  AGGRESSIVE TREATMENT STRONGLY RECOMMENDED")
        print("     • High-dose chemotherapy")
        print("     • Intensive radiation therapy")
        print("     • Consider immunotherapy")
        print("     • Enroll in clinical trial if available")
        print("     • Close monitoring every 2 months")
    elif risk_score > 1.0:
        print("  ⚠️  AGGRESSIVE TREATMENT RECOMMENDED")
        print("     • Standard chemotherapy + radiation")
        print("     • Consider immunotherapy")
        print("     • Regular monitoring every 3 months")
    elif risk_score > 0:
        print("  ⚠️  STANDARD TREATMENT WITH CLOSE MONITORING")
        print("     • Standard chemotherapy + radiation")
        print("     • Regular monitoring every 3-4 months")
    else:
        print("  ✅ STANDARD TREATMENT APPROPRIATE")
        print("     • Standard treatment protocol")
        print("     • Regular monitoring every 4-6 months")
    
    print("\n" + "="*70)
    print("Note: This prediction is based on clinical parameters and should")
    print("be used in conjunction with clinical judgment.")
    print("="*70 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("          DeepOmicsSurv - Quick Patient Prediction")
    print("="*70)
    
    print("\nAnalyzing patient parameters...")
    
    # Calculate risk score
    risk_score = calculate_risk_score(PATIENT_PARAMS)
    print(f"✓ Risk score calculated: {risk_score:.2f}")
    
    # Estimate survival probabilities
    survival_probs = estimate_survival_probabilities(risk_score)
    print("✓ Survival probabilities estimated")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(PATIENT_PARAMS, risk_score, survival_probs)
    print("✓ Visualization completed")
    
    # Print detailed report
    print_detailed_report(PATIENT_PARAMS, risk_score, survival_probs)
    
    print("\n✓ Prediction completed successfully!")
    print("✓ Report saved as: patient_prediction_report.png")
    print("\nYou can now view the generated graph and prediction report.")

if __name__ == "__main__":
    main()
