# Patient Data Folder

This folder contains sample patient data files that can be used as input for the DeepOmicsSurv prediction model **without needing the TCGA dataset**.

---

## 📁 Files Included

### JSON Format (Individual Patients)
- **patient_001_high_risk.json** - High-risk patient example
- **patient_002_low_risk.json** - Low-risk patient example
- **patient_003_moderate_risk.json** - Moderate-risk patient example
- **patient_004_very_high_risk.json** - Very high-risk patient example
- **patient_template.json** - Template for creating new patient files

### CSV Format (Multiple Patients)
- **sample_patients.csv** - 10 sample patients in CSV format

---

## 🎯 How to Use These Files

### Option 1: Use Existing Sample Patients

```python
import json

# Load a sample patient
with open('patient_data/patient_001_high_risk.json', 'r') as f:
    patient = json.load(f)

# Access clinical data
age = patient['clinical_data']['demographics']['age']
stage = patient['clinical_data']['tumor_characteristics']['tumor_stage']
print(f"Patient: Age {age}, Stage {stage}")
```

### Option 2: Create Your Own Patient File

1. Copy `patient_template.json`
2. Rename it (e.g., `patient_005.json`)
3. Fill in the clinical data
4. Save and use with prediction scripts

### Option 3: Use CSV for Batch Processing

```python
import pandas as pd

# Load all patients
patients = pd.read_csv('patient_data/sample_patients.csv')

# Process each patient
for idx, patient in patients.iterrows():
    print(f"Patient {patient['patient_id']}: Age {patient['age']}, Stage {patient['tumor_stage']}")
```

---

## 📋 Data Fields Explained

### Demographics
- **age**: Patient age in years (e.g., 65)
- **gender**: Male or Female
- **race**: White, Asian, Black, Other
- **ethnicity**: Hispanic or Latino, Not Hispanic or Latino

### Tumor Characteristics
- **primary_site**: Location of tumor
  - Oral Cavity, Oropharynx, Larynx, Hypopharynx, Other
  
- **tumor_stage**: Overall cancer stage
  - Stage I (Early, localized)
  - Stage II (Locally advanced)
  - Stage III (Advanced)
  - Stage IVA (Very advanced)
  - Stage IVB (Very advanced with extensive spread)
  - Stage IVC (Distant metastasis)
  
- **tumor_grade**: How abnormal cells look
  - G1: Well differentiated (looks more normal)
  - G2: Moderately differentiated
  - G3: Poorly differentiated (looks very abnormal)
  
- **tumor_size**: T classification
  - T1: Small tumor (≤2 cm)
  - T2: Medium tumor (2-4 cm)
  - T3: Large tumor (>4 cm)
  - T4/T4a/T4b: Very large or invasive tumor
  
- **lymph_nodes**: N classification
  - N0: No lymph node involvement
  - N1: Single lymph node (≤3 cm)
  - N2/N2a/N2b/N2c: Multiple or larger nodes
  - N3: Very large nodes (>6 cm)
  - NX: Unknown
  
- **metastasis**: M classification
  - M0: No distant metastasis
  - M1: Distant metastasis present
  - MX: Unknown

### Risk Factors
- **smoking_history**: Never, Former, Current
- **pack_years**: Number of packs per day × years smoked
- **alcohol_history**: Yes or No
- **drinks_per_week**: Number of alcoholic drinks per week
- **hpv_status**: Positive, Negative, Unknown
  - Positive is actually a good prognostic factor!

### Treatment History
- **prior_malignancy**: Previous cancer (Yes/No)
- **radiation_therapy**: Previous radiation (Yes/No)
- **chemotherapy**: Previous chemotherapy (Yes/No)
- **surgery**: Previous surgery (Yes/No)

### Vital Signs
- **weight_kg**: Weight in kilograms
- **height_cm**: Height in centimeters
- **bmi**: Body Mass Index (calculated or provided)

---

## 🎯 Sample Patients Overview

| Patient ID | Age | Gender | Stage | Grade | Smoking | Risk Category | Est. Survival |
|------------|-----|--------|-------|-------|---------|---------------|---------------|
| PATIENT-001 | 65 | Male | IVA | G3 | Former | HIGH | 18 months |
| PATIENT-002 | 45 | Female | I | G1 | Never | LOW | 72 months |
| PATIENT-003 | 58 | Male | II | G2 | Current | MODERATE | 30 months |
| PATIENT-004 | 72 | Male | IVB | G3 | Current | VERY HIGH | 10 months |
| PATIENT-005 | 50 | Female | I | G1 | Never | LOW | 60 months |
| PATIENT-006 | 62 | Male | III | G2 | Former | HIGH | 22 months |
| PATIENT-007 | 55 | Male | II | G2 | Former | MODERATE | 36 months |
| PATIENT-008 | 68 | Female | IVA | G3 | Former | HIGH | 20 months |
| PATIENT-009 | 42 | Female | I | G1 | Never | LOW | 84 months |
| PATIENT-010 | 75 | Male | IVB | G3 | Current | VERY HIGH | 8 months |

---

## 💻 Using with Prediction Scripts

### Quick Prediction Script

Create a simple script to predict from JSON:

```python
import json
import numpy as np

def load_patient_from_json(filepath):
    """Load patient data from JSON file"""
    with open(filepath, 'r') as f:
        patient = json.load(f)
    return patient

def encode_patient_data(patient):
    """Convert patient JSON to model input format"""
    clinical = patient['clinical_data']
    
    # Encode demographics
    age = clinical['demographics']['age']
    gender = 1 if clinical['demographics']['gender'] == 'Male' else 0
    race_map = {'White': 1, 'Asian': 2, 'Black': 3, 'Other': 4}
    race = race_map.get(clinical['demographics']['race'], 1)
    
    # Encode tumor characteristics
    stage_map = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 
                 'Stage IVA': 4, 'Stage IVB': 4, 'Stage IVC': 4}
    stage = stage_map.get(clinical['tumor_characteristics']['tumor_stage'], 3)
    
    grade_map = {'G1': 1, 'G2': 2, 'G3': 3}
    grade = grade_map.get(clinical['tumor_characteristics']['tumor_grade'], 2)
    
    # Encode risk factors
    smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
    smoking = smoking_map.get(clinical['risk_factors']['smoking_history'], 0)
    
    alcohol = 1 if clinical['risk_factors']['alcohol_history'] == 'Yes' else 0
    
    hpv_map = {'Negative': 0, 'Positive': 1, 'Unknown': 0}
    hpv = hpv_map.get(clinical['risk_factors']['hpv_status'], 0)
    
    # Create feature array (simplified - add more features as needed)
    features = np.array([age, gender, race, stage, grade, smoking, alcohol, hpv])
    
    return features

# Example usage
patient = load_patient_from_json('patient_data/patient_001_high_risk.json')
features = encode_patient_data(patient)
print(f"Encoded features: {features}")

# Now use these features with your trained model
# risk_score = model.predict(features)
```

---

## 📝 Creating New Patient Files

### Step 1: Copy Template
```bash
cp patient_data/patient_template.json patient_data/patient_new.json
```

### Step 2: Edit with Your Data
Open `patient_new.json` and fill in:
- Patient ID and name
- All clinical data fields
- Use exact values from the options provided

### Step 3: Validate
Make sure:
- All required fields are filled
- Values match the allowed options
- JSON syntax is correct

### Step 4: Use with Prediction
```python
patient = load_patient_from_json('patient_data/patient_new.json')
# ... run prediction
```

---

## ⚠️ Important Notes

### Genetic Data
- **Optional**: Genetic data fields are included but can be left as `null`
- **Clinical Only**: Model works with clinical data alone (slightly lower accuracy)
- **With Genetics**: If you have mRNA, methylation, or CNA data, include it for better predictions

### Data Quality
- **Accuracy**: Ensure all clinical data is accurate
- **Completeness**: Fill in all available fields
- **Consistency**: Use exact values from the options provided

### Privacy
- **De-identification**: Remove all personally identifiable information
- **Sample Data**: The provided samples are synthetic/fictional
- **Real Patients**: If using real patient data, ensure proper consent and de-identification

---

## 🔍 Field Validation

### Required Fields
- ✅ age (number)
- ✅ gender (Male/Female)
- ✅ tumor_stage (Stage I-IVC)
- ✅ tumor_grade (G1-G3)
- ✅ smoking_history (Never/Former/Current)
- ✅ hpv_status (Positive/Negative/Unknown)

### Optional Fields
- tumor_size, lymph_nodes, metastasis
- pack_years, drinks_per_week
- weight, height, bmi
- treatment history

---

## 📊 Expected Risk Categories

Based on the clinical features:

### LOW RISK (Risk Score < -0.5)
- Early stage (I-II)
- Young age (<50)
- No smoking
- HPV positive
- Well differentiated (G1)

### MODERATE RISK (Risk Score -0.5 to 1.0)
- Stage II-III
- Middle age (50-65)
- Former smoker
- Moderately differentiated (G2)

### HIGH RISK (Risk Score > 1.0)
- Advanced stage (IVA-IVB)
- Older age (>65)
- Current smoker
- HPV negative
- Poorly differentiated (G3)

---

## 🚀 Quick Start

```bash
# View sample patients
cat patient_data/sample_patients.csv

# Load a patient in Python
python3 -c "
import json
with open('patient_data/patient_001_high_risk.json') as f:
    patient = json.load(f)
    print(f\"Patient: {patient['patient_id']}\")
    print(f\"Age: {patient['clinical_data']['demographics']['age']}\")
    print(f\"Stage: {patient['clinical_data']['tumor_characteristics']['tumor_stage']}\")
"
```

---

## 📞 Need Help?

- See **COMPLETE_USER_GUIDE.md** for full documentation
- See **QUICK_REFERENCE.md** for quick commands
- See **START_HERE.md** for navigation

---

**Last Updated:** December 4, 2024
