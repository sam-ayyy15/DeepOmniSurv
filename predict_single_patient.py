"""
Simple example: How to predict survival for a single patient
This demonstrates the input format and prediction process
"""

import numpy as np
import pandas as pd
from deepomicsurv_implementation import DataLoader, DeepOmicsSurv
from sklearn.preprocessing import StandardScaler

def predict_patient_survival():
    """
    Example: Predict survival for a new patient
    """
    
    print("=" * 60)
    print("SINGLE PATIENT SURVIVAL PREDICTION EXAMPLE")
    print("=" * 60)
    
    # Step 1: Load the trained model and data
    print("\n1. Loading data and preprocessing...")
    data_loader = DataLoader()
    data_loader.load_tcga_data()
    X_clinical, y_time, y_event = data_loader.preprocess_clinical_data()
    
    # Step 2: Train a simple model (in practice, you'd load a pre-trained model)
    print("\n2. Training model (in practice, load pre-trained model)...")
    from sklearn.model_selection import train_test_split
    y_survival = np.column_stack([y_time, y_event])
    X_train, X_test, y_train, y_test = train_test_split(
        X_clinical, y_survival, test_size=0.3, random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = DeepOmicsSurv(input_dim=X_train_scaled.shape[1])
    model.build_model()
    model.compile_model()
    model.train(X_train_scaled, y_train, epochs=50, verbose=0)
    
    print("Model trained successfully!")
    
    # Step 3: Create a sample new patient
    print("\n3. Creating sample patient data...")
    print("\n" + "=" * 60)
    print("SAMPLE PATIENT INPUT")
    print("=" * 60)
    
    # Get feature names
    feature_names = data_loader.processed_data['clinical']['feature_names']
    
    # Take the first test patient as an example
    sample_patient = X_test[0:1]  # Shape: (1, n_features)
    actual_survival_time = y_test[0, 0]
    actual_event = y_test[0, 1]
    
    # Display patient information
    print("\nPatient Clinical Features:")
    print("-" * 60)
    for i, feature_name in enumerate(feature_names[:10]):  # Show first 10 features
        print(f"  {feature_name}: {sample_patient[0, i]:.2f}")
    print(f"  ... and {len(feature_names) - 10} more features")
    
    print(f"\nActual Outcome:")
    print(f"  Survival Time: {actual_survival_time:.1f} months")
    print(f"  Event (Death): {'Yes' if actual_event == 1 else 'No (Censored)'}")
    
    # Step 4: Make prediction
    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)
    
    # Scale the patient data
    sample_patient_scaled = scaler.transform(sample_patient)
    
    # Predict risk score
    risk_score = model.predict(sample_patient_scaled)[0]
    
    print(f"\nPredicted Risk Score: {risk_score:.4f}")
    print("\nInterpretation:")
    print(f"  - Higher score = Higher risk of death")
    print(f"  - Lower score = Lower risk of death")
    
    # Compare with other patients
    all_predictions = model.predict(X_test_scaled)
    percentile = (all_predictions < risk_score).sum() / len(all_predictions) * 100
    
    print(f"\nRisk Percentile: {percentile:.1f}%")
    print(f"  (This patient is at higher risk than {percentile:.1f}% of patients)")
    
    # Risk classification
    median_risk = np.median(all_predictions)
    if risk_score > median_risk:
        risk_group = "HIGH RISK"
        print(f"\nRisk Group: {risk_group}")
        print("  Recommendation: Close monitoring, aggressive treatment")
    else:
        risk_group = "LOW RISK"
        print(f"\nRisk Group: {risk_group}")
        print("  Recommendation: Standard follow-up protocol")
    
    # Step 5: Show how to format input for a completely new patient
    print("\n" + "=" * 60)
    print("HOW TO FORMAT INPUT FOR A NEW PATIENT")
    print("=" * 60)
    
    print("\nRequired Input Format:")
    print("-" * 60)
    print("You need a numpy array with shape (1, n_features)")
    print(f"For this model: (1, {len(feature_names)}) features")
    print("\nFeatures include:")
    print("  - Numerical: age, tumor measurements, etc.")
    print("  - Encoded categorical: gender, race, tumor stage, etc.")
    print("\nExample code:")
    print("""
    # Create patient data array
    new_patient = np.array([[
        65.0,  # age
        1,     # gender_encoded (0=Female, 1=Male)
        2,     # race_encoded
        4,     # tumor_stage_encoded (1=Stage I, 4=Stage IV)
        # ... more features
    ]])
    
    # Scale the data
    new_patient_scaled = scaler.transform(new_patient)
    
    # Predict
    risk_score = model.predict(new_patient_scaled)[0]
    print(f"Risk Score: {risk_score:.4f}")
    """)
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED!")
    print("=" * 60)
    
    return model, scaler, feature_names

if __name__ == "__main__":
    model, scaler, feature_names = predict_patient_survival()
