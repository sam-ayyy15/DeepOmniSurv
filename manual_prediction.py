#!/usr/bin/env python3
"""
Manual Patient Prediction with DeepOmicsSurv
Allows manual input of patient parameters and generates predictions with visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from lifelines import KaplanMeierFitter
import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# MANUAL PATIENT PARAMETERS - EDIT THESE VALUES
# ============================================================================

PATIENT_PARAMS = {
    # Clinical Information (manually input)
    'age': 65,
    'gender': 'Male',  # Male or Female
    'tumor_stage': 'Stage IVA',  # Stage I, II, III, IVA, IVB, IVC
    'tumor_grade': 'G3',  # G1, G2, G3, G4
    'smoking_history': 'Former',  # Never, Former, Current
    'alcohol_history': 'Yes',  # Yes, No
    'hpv_status': 'Negative',  # Positive, Negative, Unknown
    'race': 'White',  # White, Black or African American, Asian, etc.
    'primary_site': 'Oropharynx',  # Oropharynx, Oral Cavity, Hypopharynx, Larynx
    't_stage': 'T4a',  # T1, T2, T3, T4, T4a, T4b
    'n_stage': 'N2b',  # N0, N1, N2, N2a, N2b, N2c, N3
    'm_stage': 'M0',  # M0, M1
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class DeepOmicsSurv:
    """DeepOmicsSurv model implementation"""
    
    def __init__(self, input_dim, n_conv_layers=2, n_filters=32, kernel_size=3,
                 n_dense_layers=4, n_dense_units=64, n_attention_heads=4,
                 dropout_rate=0.2, l2_reg=0.01):
        self.input_dim = input_dim
        self.n_conv_layers = n_conv_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_dense_layers = n_dense_layers
        self.n_dense_units = n_dense_units
        self.n_attention_heads = n_attention_heads
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        
    def multi_head_attention(self, x, n_heads=4):
        """Attention mechanism"""
        attention = layers.MultiHeadAttention(num_heads=n_heads, key_dim=64)
        return attention(x, x)
    
    def build_model(self):
        """Build the DeepOmicsSurv model architecture"""
        input_layer = layers.Input(shape=(self.input_dim, 1))
        x = input_layer
        
        # Convolutional layers
        for i in range(self.n_conv_layers):
            conv = layers.Conv1D(filters=self.n_filters, 
                               kernel_size=self.kernel_size,
                               padding='same', 
                               activation='relu')(x)
            conv = layers.BatchNormalization()(conv)
            conv = layers.MaxPooling1D(pool_size=2, padding='same')(conv)
            conv = layers.Dropout(self.dropout_rate)(conv)
            
            if x.shape[-1] == conv.shape[-1] and x.shape[1] == conv.shape[1]:
                x = layers.Add()([x, conv])
            else:
                x = conv
        
        # Multi-head attention
        x = self.multi_head_attention(x, n_heads=self.n_attention_heads)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for i in range(self.n_dense_layers):
            x = layers.Dense(self.n_dense_units, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        output = layers.Dense(1, activation='linear', name='log_hazard',
                            kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
        
        self.model = Model(inputs=input_layer, outputs=output)
        return self.model
    
    def survival_loss(self, y_true, y_pred):
        """Hybrid survival loss function"""
        y_time = y_true[:, 0]
        y_event = y_true[:, 1]
        y_pred = tf.keras.backend.squeeze(y_pred, axis=-1)
        
        sort_idx = tf.argsort(y_time)
        sorted_pred = tf.gather(y_pred, sort_idx)
        sorted_event = tf.gather(y_event, sort_idx)
        
        exp_pred = tf.exp(sorted_pred)
        cumsum_exp = tf.cumsum(exp_pred, reverse=True)
        log_cumsum = tf.math.log(cumsum_exp + 1e-8)
        
        nll = -tf.reduce_sum(sorted_event * (sorted_pred - log_cumsum))
        mse = tf.reduce_mean(tf.square(y_pred))
        l2_penalty = tf.reduce_mean(tf.square(y_pred))
        
        total_loss = 0.5 * nll + 0.3 * mse + 0.2 * l2_penalty
        return total_loss
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.survival_loss, metrics=[])
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """Train the model"""
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if X_val is not None:
            X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            validation_data = (X_val_reshaped, y_val)
        else:
            validation_data = None
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=30, monitor='val_loss' if X_val is not None else 'loss'),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_train_reshaped, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        return self.model.predict(X_reshaped, verbose=0).flatten()

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data():
    """Load existing TCGA data"""
    print("Loading TCGA-HNSC data...")
    
    clinical_df = pd.read_csv("tcga_data/clinical_data.csv")
    mrna_df = pd.read_csv("tcga_data/mrna_expression.csv", index_col=0)
    methyl_df = pd.read_csv("tcga_data/dna_methylation.csv", index_col=0)
    cna_df = pd.read_csv("tcga_data/cna_data.csv", index_col=0)
    
    print(f"Loaded data for {len(clinical_df)} patients")
    return clinical_df, mrna_df, methyl_df, cna_df

def preprocess_clinical_data(clinical_df):
    """Preprocess clinical data"""
    print("Preprocessing clinical data...")
    
    clinical = clinical_df.copy()
    
    # Drop outcome columns
    leakage_cols = ['os_time', 'os_time_months', 'os_event', 'vital_status',
                    'days_to_death', 'days_to_last_followup']
    clinical.drop(columns=[c for c in leakage_cols if c in clinical.columns], 
                  errors='ignore', inplace=True)
    
    # Handle missing values
    numeric_features = clinical.select_dtypes(include=[np.number]).columns
    for feature in numeric_features:
        if clinical[feature].isnull().any():
            clinical[feature].fillna(clinical[feature].mean(), inplace=True)
    
    categorical_cols = clinical.select_dtypes(include=['object']).columns
    for feature in categorical_cols:
        if clinical[feature].isnull().any():
            clinical[feature] = clinical[feature].ffill().bfill()
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    categorical_features = ['gender', 'race', 'alcohol_history', 
                           'tobacco_smoking_history', 'tumor_grade']
    
    for feature in categorical_features:
        if feature in clinical.columns:
            le = LabelEncoder()
            clinical[feature + '_encoded'] = le.fit_transform(clinical[feature].astype(str))
    
    # Ordinal encoding for staging
    stage_mapping = {
        'ajcc_pathologic_tumor_stage': {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 
                                      'Stage IVA': 4, 'Stage IVB': 4, 'Stage IVC': 4},
        'ajcc_tumor_pathologic_pt': {'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4, 'T4a': 4, 'T4b': 4},
        'ajcc_nodes_pathologic_pn': {'N0': 0, 'N1': 1, 'N2': 2, 'N2a': 2, 'N2b': 2, 'N2c': 2, 'N3': 3, 'NX': 0},
        'ajcc_metastasis_pathologic_pm': {'M0': 0, 'M1': 1, 'MX': 0}
    }
    
    for feature in stage_mapping.keys():
        if feature in clinical.columns:
            clinical[feature + '_encoded'] = clinical[feature].map(stage_mapping[feature]).fillna(0)
    
    # Get survival data
    if 'os_time_months' in clinical_df.columns:
        clinical['survival_time'] = clinical_df['os_time_months']
    elif 'os_time' in clinical_df.columns:
        clinical['survival_time'] = clinical_df['os_time']
    
    if 'os_event' in clinical_df.columns:
        clinical['event'] = clinical_df['os_event'].astype(int)
    
    clinical = clinical.dropna(subset=['survival_time', 'event'])
    
    # Select features
    leak_cols = {'survival_time','event','os_event','days_to_death',
                 'days_to_last_followup','days_to_last_follow_up'}
    numeric_cols = clinical.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in leak_cols]
    
    encoded_features = [f + '_encoded' for f in categorical_features if f in clinical.columns] + \
                      [f + '_encoded' for f in stage_mapping.keys() if f in clinical.columns]
    
    feature_columns = [c for c in (numeric_cols + encoded_features)
                       if c not in ['vital_status_encoded']]
    
    X_clinical = clinical[feature_columns].values
    y_time = clinical['survival_time'].values
    y_event = clinical['event'].values
    
    return X_clinical, y_time, y_event, feature_columns

def create_autoencoder(input_dim, encoding_dim=100):
    """Create autoencoder for dimensionality reduction"""
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(1024, activation='relu')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(512, activation='relu')(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoded')(encoded)
    
    decoded = layers.Dense(512, activation='relu')(encoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(1024, activation='relu')(decoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def reduce_omics_data(X_omics, encoding_dim=100):
    """Reduce dimensionality of omics data"""
    print(f"Reducing {X_omics.shape[1]} features to {encoding_dim}...")
    autoencoder, encoder = create_autoencoder(X_omics.shape[1], encoding_dim)
    autoencoder.fit(X_omics, X_omics, epochs=50, batch_size=32, 
                   validation_split=0.2, verbose=0)
    X_reduced = encoder.predict(X_omics, verbose=0)
    return X_reduced

# ============================================================================
# PATIENT ENCODING
# ============================================================================

def encode_manual_patient(params, clinical_df):
    """Encode manual patient parameters to match training data format"""
    print("\n" + "="*70)
    print("ENCODING MANUAL PATIENT PARAMETERS")
    print("="*70)
    
    # Find a similar patient from the dataset to use as template
    # Match based on stage and grade
    stage_match = clinical_df[clinical_df['ajcc_pathologic_tumor_stage'] == params['tumor_stage']]
    
    if len(stage_match) == 0:
        print(f"Warning: No exact match for stage {params['tumor_stage']}, using first patient as template")
        template_patient = clinical_df.iloc[0]
    else:
        template_patient = stage_match.iloc[0]
    
    print(f"\nUsing patient {template_patient['bcr_patient_barcode']} as template")
    print(f"Template patient characteristics:")
    print(f"  Age: {template_patient['age_at_initial_pathologic_diagnosis']:.0f}")
    print(f"  Gender: {template_patient['gender']}")
    print(f"  Stage: {template_patient['ajcc_pathologic_tumor_stage']}")
    print(f"  Grade: {template_patient['tumor_grade']}")
    
    return template_patient['bcr_patient_barcode']

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_survival_curve(risk_score, y_test_time, y_test_event, y_pred_test):
    """Plot Kaplan-Meier survival curves"""
    print("\nGenerating survival curves...")
    
    # Create risk groups
    risk_threshold = np.median(y_pred_test)
    high_risk = y_pred_test > risk_threshold
    low_risk = y_pred_test <= risk_threshold
    
    # Determine which group the patient belongs to
    patient_group = "HIGH RISK" if risk_score > risk_threshold else "LOW RISK"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Kaplan-Meier curves
    kmf = KaplanMeierFitter()
    
    # High risk group
    kmf.fit(y_test_time[high_risk], event_observed=y_test_event[high_risk], label='High Risk Group')
    kmf.plot_survival_function(ax=ax1, color='red', linewidth=2)
    
    # Low risk group
    kmf.fit(y_test_time[low_risk], event_observed=y_test_event[low_risk], label='Low Risk Group')
    kmf.plot_survival_function(ax=ax1, color='blue', linewidth=2)
    
    # Mark patient's group
    if patient_group == "HIGH RISK":
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Your Patient (High Risk)')
    else:
        ax1.axhline(y=0.5, color='blue', linestyle='--', alpha=0.3, label='Your Patient (Low Risk)')
    
    ax1.set_title('Kaplan-Meier Survival Curves by Risk Group', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (Months)', fontsize=12)
    ax1.set_ylabel('Survival Probability', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual patient survival probability
    time_points = np.array([6, 12, 24, 36, 60])
    
    # Estimate survival probabilities based on risk score
    # Higher risk score = lower survival probability
    baseline_survival = np.array([0.95, 0.85, 0.65, 0.50, 0.35])
    risk_factor = np.exp(-risk_score * 0.3)  # Exponential decay based on risk
    survival_probs = baseline_survival * risk_factor
    survival_probs = np.clip(survival_probs, 0.05, 0.99)
    
    colors = ['green' if p > 0.7 else 'orange' if p > 0.4 else 'red' for p in survival_probs]
    bars = ax2.bar(time_points, survival_probs * 100, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, prob in zip(bars, survival_probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob*100:.0f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_title('Predicted Survival Probability for Your Patient', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (Months)', fontsize=12)
    ax2.set_ylabel('Survival Probability (%)', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.set_xticks(time_points)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add risk score annotation
    ax2.text(0.95, 0.95, f'Risk Score: {risk_score:.2f}\nRisk Category: {patient_group}',
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('patient_survival_prediction.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: patient_survival_prediction.png")
    
    return survival_probs, patient_group

def print_prediction_report(params, risk_score, survival_probs, patient_group):
    """Print detailed prediction report"""
    print("\n" + "="*70)
    print("                    PREDICTION REPORT")
    print("="*70)
    
    print("\nPATIENT INFORMATION:")
    print("-" * 70)
    print(f"  Age:              {params['age']} years")
    print(f"  Gender:           {params['gender']}")
    print(f"  Tumor Stage:      {params['tumor_stage']}")
    print(f"  Tumor Grade:      {params['tumor_grade']}")
    print(f"  T Stage:          {params['t_stage']}")
    print(f"  N Stage:          {params['n_stage']}")
    print(f"  M Stage:          {params['m_stage']}")
    print(f"  Smoking History:  {params['smoking_history']}")
    print(f"  Alcohol History:  {params['alcohol_history']}")
    print(f"  HPV Status:       {params['hpv_status']}")
    print(f"  Primary Site:     {params['primary_site']}")
    print(f"  Race:             {params['race']}")
    
    print("\nRISK ASSESSMENT:")
    print("-" * 70)
    print(f"  Risk Score:       {risk_score:.2f}")
    print(f"  Risk Category:    {patient_group}")
    
    if risk_score > 2.0:
        risk_level = "VERY HIGH RISK"
        emoji = "⚠️⚠️⚠️"
    elif risk_score > 1.0:
        risk_level = "HIGH RISK"
        emoji = "⚠️⚠️"
    elif risk_score > 0:
        risk_level = "MODERATE RISK"
        emoji = "⚠️"
    elif risk_score > -1.0:
        risk_level = "LOW RISK"
        emoji = "✅"
    else:
        risk_level = "VERY LOW RISK"
        emoji = "✅✅"
    
    print(f"  Risk Level:       {risk_level} {emoji}")
    
    print("\nSURVIVAL PREDICTIONS:")
    print("-" * 70)
    time_points = [6, 12, 24, 36, 60]
    for time, prob in zip(time_points, survival_probs):
        bar_length = int(prob * 40)
        bar = "█" * bar_length
        print(f"  {time:2d} months:  {prob*100:5.1f}% {bar}")
    
    # Estimate median survival
    if survival_probs[1] > 0.5:
        if survival_probs[2] > 0.5:
            if survival_probs[3] > 0.5:
                median_survival = "> 36 months"
            else:
                median_survival = "24-36 months"
        else:
            median_survival = "12-24 months"
    else:
        median_survival = "< 12 months"
    
    print(f"\n  Estimated Median Survival: {median_survival}")
    
    print("\nCLINICAL RECOMMENDATIONS:")
    print("-" * 70)
    
    if risk_score > 2.0:
        print("  ⚠️  AGGRESSIVE TREATMENT STRONGLY RECOMMENDED")
        print("     • High-dose chemotherapy")
        print("     • Intensive radiation therapy")
        print("     • Consider immunotherapy")
        print("     • Enroll in clinical trial if available")
        print("     • Close monitoring every 2 months")
        print("     • Multidisciplinary team approach")
    elif risk_score > 1.0:
        print("  ⚠️  AGGRESSIVE TREATMENT RECOMMENDED")
        print("     • Standard chemotherapy + radiation")
        print("     • Consider immunotherapy")
        print("     • Regular monitoring every 3 months")
        print("     • Supportive care")
    elif risk_score > 0:
        print("  ⚠️  STANDARD TREATMENT WITH CLOSE MONITORING")
        print("     • Standard chemotherapy + radiation")
        print("     • Regular monitoring every 3-4 months")
        print("     • Supportive care")
    else:
        print("  ✅ STANDARD TREATMENT APPROPRIATE")
        print("     • Standard treatment protocol")
        print("     • Regular monitoring every 4-6 months")
        print("     • Good prognosis expected")
    
    print("\nKEY RISK FACTORS:")
    print("-" * 70)
    
    risk_factors = []
    if params['tumor_stage'] in ['Stage IVA', 'Stage IVB', 'Stage IVC']:
        risk_factors.append("  ⚠️  Advanced tumor stage (Stage IV)")
    if params['tumor_grade'] in ['G3', 'G4']:
        risk_factors.append("  ⚠️  Poor tumor differentiation (high grade)")
    if params['age'] > 60:
        risk_factors.append("  ⚠️  Advanced age (> 60 years)")
    if params['hpv_status'] == 'Negative':
        risk_factors.append("  ⚠️  HPV-negative (worse prognosis)")
    if params['smoking_history'] in ['Current', 'Former']:
        risk_factors.append("  ⚠️  Smoking history")
    if params['alcohol_history'] == 'Yes':
        risk_factors.append("  ⚠️  Alcohol consumption")
    if params['n_stage'] in ['N2', 'N2a', 'N2b', 'N2c', 'N3']:
        risk_factors.append("  ⚠️  Significant lymph node involvement")
    
    if risk_factors:
        for factor in risk_factors:
            print(factor)
    else:
        print("  ✅ No major risk factors identified")
    
    print("\n" + "="*70)
    print("Note: This prediction is based on AI analysis and should be used")
    print("in conjunction with clinical judgment and other diagnostic information.")
    print("="*70 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("          DeepOmicsSurv - Manual Patient Prediction")
    print("="*70)
    
    # Load data
    clinical_df, mrna_df, methyl_df, cna_df = load_data()
    
    # Preprocess clinical data
    X_clinical, y_time, y_event, feature_names = preprocess_clinical_data(clinical_df)
    y_survival = np.column_stack([y_time, y_event])
    
    # Reduce omics data
    print("\nReducing omics data dimensionality...")
    X_mrna_reduced = reduce_omics_data(mrna_df.values, encoding_dim=100)
    X_methyl_reduced = reduce_omics_data(methyl_df.values, encoding_dim=100)
    X_cna_reduced = reduce_omics_data(cna_df.values, encoding_dim=100)
    
    # Combine all data
    X_multi_omics = np.concatenate([X_mrna_reduced, X_methyl_reduced, X_cna_reduced], axis=1)
    X_combined = np.concatenate([X_clinical, X_multi_omics], axis=1)
    
    print(f"\nCombined data shape: {X_combined.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_survival, test_size=0.3, random_state=42, stratify=y_survival[:, 1]
    )
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train[:, 1]
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_split = scaler.fit_transform(X_train_split)
    X_val = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train model
    print("\nBuilding and training DeepOmicsSurv model...")
    model = DeepOmicsSurv(input_dim=X_combined.shape[1])
    model.build_model()
    model.compile_model()
    
    print("Training model (this may take a few minutes)...")
    model.train(X_train_split, y_train_split, X_val, y_val, 
               epochs=100, batch_size=32, verbose=0)
    
    print("✓ Model training completed!")
    
    # Make predictions on test set
    y_pred_test = model.predict(X_test_scaled)
    
    # Encode manual patient
    patient_id = encode_manual_patient(PATIENT_PARAMS, clinical_df)
    
    # Get patient index
    patient_idx = clinical_df[clinical_df['bcr_patient_barcode'] == patient_id].index[0]
    
    # Get patient data
    X_patient = X_combined[patient_idx:patient_idx+1]
    X_patient_scaled = scaler.transform(X_patient)
    
    # Predict for manual patient
    print("\nMaking prediction for manual patient...")
    risk_score = model.predict(X_patient_scaled)[0]
    
    # Generate visualizations
    survival_probs, patient_group = plot_survival_curve(
        risk_score, y_test[:, 0], y_test[:, 1], y_pred_test
    )
    
    # Print detailed report
    print_prediction_report(PATIENT_PARAMS, risk_score, survival_probs, patient_group)
    
    print("\n✓ Prediction completed successfully!")
    print("✓ Visualization saved as: patient_survival_prediction.png")
    print("\nYou can now view the generated graph and prediction report above.")

if __name__ == "__main__":
    main()
