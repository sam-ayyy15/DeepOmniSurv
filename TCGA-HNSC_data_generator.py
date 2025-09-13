#!/usr/bin/env python3
"""
Realistic TCGA-HNSC Data Generator
Creates datasets that closely match the real TCGA-HNSC data characteristics
Based on the paper's Table 1 and typical TCGA data properties
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealisticTCGAGenerator:
    """Generate realistic TCGA-HNSC datasets matching paper specifications"""
    
    def __init__(self, n_patients=528, random_seed=42):
        self.n_patients = n_patients
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Create patient IDs in TCGA format
        self.patient_ids = [f"TCGA-{chr(65 + i//100)}{chr(65 + (i//10)%10)}-{i%100:04d}" 
                           for i in range(n_patients)]
        
        # Barcode IDs for samples (TCGA format)
        self.sample_ids = [f"{pid}-01A-11R-1234-07" for pid in self.patient_ids]
        
    def generate_clinical_data(self):
        """Generate clinical data based on Table 1 from the paper"""
        print("Generating realistic clinical data...")
        
        # Gender distribution from paper: Male 73.11%, Female 26.89%
        gender = np.random.choice(['Male', 'Female'], self.n_patients, p=[0.7311, 0.2689])
        
        # Age distribution from paper
        age_ranges = ['<30', '31-40', '41-50', '51-60', '>60']
        age_probs = [0.0152, 0.0265, 0.1420, 0.3087, 0.5076]
        age_categories = np.random.choice(age_ranges, self.n_patients, p=age_probs)
        
        # Convert age categories to actual ages
        age_at_diagnosis = []
        for cat in age_categories:
            if cat == '<30':
                age = np.random.uniform(18, 29)
            elif cat == '31-40':
                age = np.random.uniform(31, 40)
            elif cat == '41-50':
                age = np.random.uniform(41, 50)
            elif cat == '51-60':
                age = np.random.uniform(51, 60)
            else:  # >60
                age = np.random.uniform(61, 85)
            age_at_diagnosis.append(age)
        
        # Race distribution from paper
        race_options = ['White', 'Black or African American', 'Asian', 'American Indian or Alaska Native']
        race_probs = [0.8561, 0.0909, 0.0208, 0.0038]
        # Normalize probabilities to sum to 1
        race_probs = np.array(race_probs) / np.sum(race_probs)
        race = np.random.choice(race_options, self.n_patients, p=race_probs)
        
        # AJCC Clinical Stage from paper
        stage_options = ['Stage I', 'Stage II', 'Stage III', 'Stage IVA', 'Stage IVB', 'Stage IVC']
        stage_probs = [0.0398, 0.1875, 0.2027, 0.5095, 0.0208, 0.0133]
        # Normalize probabilities to sum to 1
        stage_probs = np.array(stage_probs) / np.sum(stage_probs)
        ajcc_clinical_stage = np.random.choice(stage_options, self.n_patients, p=stage_probs)
        
        # AJCC Clinical T
        t_options = ['T1', 'T2', 'T3', 'T4', 'T4a', 'T4b']
        t_probs = [0.0701, 0.2879, 0.2633, 0.0473, 0.2955, 0.0057]
        # Normalize probabilities to sum to 1
        t_probs = np.array(t_probs) / np.sum(t_probs)
        ajcc_clinical_t = np.random.choice(t_options, self.n_patients, p=t_probs)
        
        # AJCC Clinical N
        n_options = ['N0', 'N1', 'N2', 'N2a', 'N2b', 'N2c', 'N3', 'NX']
        n_probs = [0.4659, 0.1610, 0.0360, 0.0322, 0.1610, 0.0852, 0.0170, 0.0341]
        # Normalize probabilities to sum to 1
        n_probs = np.array(n_probs) / np.sum(n_probs)
        ajcc_clinical_n = np.random.choice(n_options, self.n_patients, p=n_probs)
        
        # AJCC Clinical M
        m_options = ['M0', 'M1', 'MX']
        m_probs = [0.9394, 0.0114, 0.0398]
        # Normalize probabilities to sum to 1
        m_probs = np.array(m_probs) / np.sum(m_probs)
        ajcc_clinical_m = np.random.choice(m_options, self.n_patients, p=m_probs)
        
        # Alcohol History from paper
        alcohol_options = ['Yes', 'No', 'Unknown']
        alcohol_probs = [0.6667, 0.3125, 0.0208]
        alcohol_history = np.random.choice(alcohol_options, self.n_patients, p=alcohol_probs)
        
        # Smoking History (derived from typical HNSC patterns)
        smoking_options = ['Current', 'Former', 'Never', 'Unknown']
        smoking_probs = [0.25, 0.45, 0.25, 0.05]
        smoking_history = np.random.choice(smoking_options, self.n_patients, p=smoking_probs)
        
        # HPV Status (important for HNSC)
        hpv_status = np.random.choice(['Positive', 'Negative', 'Unknown'], 
                                     self.n_patients, p=[0.25, 0.65, 0.10])
        
        # Primary Site
        primary_sites = ['Oropharynx', 'Oral Cavity', 'Hypopharynx', 'Larynx', 'Other']
        site_probs = [0.35, 0.30, 0.15, 0.15, 0.05]
        primary_site = np.random.choice(primary_sites, self.n_patients, p=site_probs)
        
        # Generate realistic survival data
        survival_data = self._generate_survival_data(
            age_at_diagnosis, ajcc_clinical_stage, ajcc_clinical_t, 
            ajcc_clinical_n, smoking_history, alcohol_history, hpv_status
        )
        
        # Create clinical dataframe
        clinical_df = pd.DataFrame({
            'bcr_patient_barcode': self.patient_ids,
            'age_at_initial_pathologic_diagnosis': age_at_diagnosis,
            'gender': gender,
            'race': race,
            'ethnicity': np.random.choice(['Hispanic or Latino', 'Not Hispanic or Latino', 'Unknown'], 
                                        self.n_patients, p=[0.05, 0.90, 0.05]),
            'ajcc_pathologic_tumor_stage': ajcc_clinical_stage,
            'ajcc_tumor_pathologic_pt': ajcc_clinical_t,
            'ajcc_nodes_pathologic_pn': ajcc_clinical_n,
            'ajcc_metastasis_pathologic_pm': ajcc_clinical_m,
            'alcohol_history': alcohol_history,
            'tobacco_smoking_history': smoking_history,
            'hpv_status': hpv_status,
            'primary_site': primary_site,
            'tumor_grade': np.random.choice(['G1', 'G2', 'G3', 'G4', 'GX'], 
                                          self.n_patients, p=[0.05, 0.30, 0.45, 0.15, 0.05]),
            'prior_malignancy': np.random.choice(['Yes', 'No'], self.n_patients, p=[0.15, 0.85]),
            'radiation_therapy': np.random.choice(['Yes', 'No'], self.n_patients, p=[0.70, 0.30]),
            'pharmaceutical_therapy_type': np.random.choice(['Chemotherapy', 'Targeted Therapy', 'None'], 
                                                          self.n_patients, p=[0.60, 0.20, 0.20]),
            **survival_data
        })
        
        return clinical_df
    
    def _generate_survival_data(self, ages, stages, t_stages, n_stages, smoking, alcohol, hpv):
        """Generate realistic survival data with proper relationships"""
        
        # Create risk scores based on clinical factors
        risk_scores = np.zeros(self.n_patients)
        
        for i in range(self.n_patients):
            risk = 0
            
            # Age effect
            risk += (ages[i] - 50) * 0.02
            
            # Stage effect
            if stages[i] in ['Stage III', 'Stage IVA']:
                risk += 1.0
            elif stages[i] in ['Stage IVB', 'Stage IVC']:
                risk += 1.8
            
            # T stage effect
            if t_stages[i] in ['T3', 'T4', 'T4a', 'T4b']:
                risk += 0.5
            
            # N stage effect
            if n_stages[i] in ['N2', 'N2a', 'N2b', 'N2c', 'N3']:
                risk += 0.8
            
            # Smoking effect
            if smoking[i] in ['Current', 'Former']:
                risk += 0.3
            
            # Alcohol effect
            if alcohol[i] == 'Yes':
                risk += 0.2
            
            # HPV effect (protective for oropharyngeal)
            if hpv[i] == 'Positive':
                risk -= 0.5
            
            risk_scores[i] = max(0.1, risk)
        
        # Generate survival times using Weibull distribution
        # Scale parameter inversely related to risk
        scale_params = 2000 / (risk_scores + 1)  # Days
        shape_param = 1.5  # Weibull shape parameter
        
        survival_times = np.random.weibull(shape_param, self.n_patients) * scale_params
        survival_times = np.clip(survival_times, 1, 5000)  # 1 day to ~13.7 years
        
        # Generate event indicators (death = 1, alive = 0)
        # Higher risk = higher probability of death
        death_probs = 1 / (1 + np.exp(-(risk_scores - 1)))  # Sigmoid function
        vital_status = np.random.binomial(1, death_probs)
        
        # For alive patients, generate follow-up times
        days_to_death = np.where(vital_status == 1, survival_times, np.nan)
        days_to_last_followup = np.where(vital_status == 0, 
                                       survival_times + np.random.exponential(500, self.n_patients), 
                                       np.nan)
        
        # Create overall survival time (key variable for analysis)
        os_time = np.where(vital_status == 1, days_to_death, days_to_last_followup)
        os_event = vital_status
        
        return {
            'vital_status': np.where(vital_status == 1, 'Dead', 'Alive'),
            'days_to_death': days_to_death,
            'days_to_last_followup': days_to_last_followup,
            'os_time': os_time,  # Overall survival time in days
            'os_event': os_event,  # Event indicator (1=death, 0=censored)
            'os_time_months': os_time / 30.44  # Convert to months
        }
    
    def generate_mrna_data(self):
        """Generate realistic mRNA expression data"""
        print("Generating mRNA expression data (20,531 genes)...")
        
        # Use real gene symbols for realism
        gene_symbols = self._get_gene_symbols(20531)
        
        # Generate log2(TPM+1) values typical for RNA-seq
        # Most genes have low expression, some have high expression
        mrna_data = np.zeros((self.n_patients, len(gene_symbols)))
        
        for i, gene in enumerate(gene_symbols):
            if i % 5000 == 0:
                print(f"Processing gene {i}/{len(gene_symbols)}")
            
            # Different expression patterns for different genes
            if np.random.random() < 0.1:  # 10% high expression genes
                base_expr = np.random.lognormal(3, 1, self.n_patients)
            elif np.random.random() < 0.3:  # 30% moderate expression
                base_expr = np.random.lognormal(1, 1, self.n_patients)
            else:  # 60% low expression
                base_expr = np.random.lognormal(-1, 1.5, self.n_patients)
            
            # Add some patient-specific variation
            patient_effects = np.random.normal(0, 0.2, self.n_patients)
            mrna_data[:, i] = np.log2(base_expr + patient_effects + 1)
            mrna_data[:, i] = np.clip(mrna_data[:, i], 0, 20)  # Clip to realistic range
        
        # Create DataFrame
        mrna_df = pd.DataFrame(mrna_data, 
                              index=self.patient_ids, 
                              columns=gene_symbols)
        mrna_df.index.name = 'bcr_patient_barcode'
        
        return mrna_df
    
    def generate_methylation_data(self):
        """Generate realistic DNA methylation data"""
        print("Generating DNA methylation data (16,529 CpG sites)...")
        
        # Generate CpG probe IDs in Illumina format
        cpg_probes = [f"cg{i:08d}" for i in range(16529)]
        
        # Generate beta values (0-1 range for methylation)
        methyl_data = np.zeros((self.n_patients, len(cpg_probes)))
        
        for i, probe in enumerate(cpg_probes):
            if i % 3000 == 0:
                print(f"Processing probe {i}/{len(cpg_probes)}")
            
            # Different methylation patterns
            if np.random.random() < 0.15:  # 15% highly methylated probes
                methyl_data[:, i] = np.random.beta(8, 2, self.n_patients)
            elif np.random.random() < 0.15:  # 15% lowly methylated probes  
                methyl_data[:, i] = np.random.beta(2, 8, self.n_patients)
            else:  # 70% intermediate methylation
                methyl_data[:, i] = np.random.beta(3, 3, self.n_patients)
        
        # Clip to valid range
        methyl_data = np.clip(methyl_data, 0, 1)
        
        # Create DataFrame
        methyl_df = pd.DataFrame(methyl_data, 
                                index=self.patient_ids, 
                                columns=cpg_probes)
        methyl_df.index.name = 'bcr_patient_barcode'
        
        return methyl_df
    
    def generate_cna_data(self):
        """Generate realistic Copy Number Alteration data"""
        print("Generating Copy Number Alteration data (24,776 genes)...")
        
        # Use gene symbols
        gene_symbols = self._get_gene_symbols(24776)
        
        # Generate copy number values (log2 ratio, typically -2 to +2)
        cna_data = np.zeros((self.n_patients, len(gene_symbols)))
        
        for i, gene in enumerate(gene_symbols):
            if i % 5000 == 0:
                print(f"Processing gene {i}/{len(gene_symbols)}")
            
            # Most regions are neutral (log2 ratio ~ 0)
            if np.random.random() < 0.05:  # 5% amplified regions
                cna_data[:, i] = np.random.normal(1.0, 0.5, self.n_patients)
            elif np.random.random() < 0.05:  # 5% deleted regions
                cna_data[:, i] = np.random.normal(-1.0, 0.5, self.n_patients)
            else:  # 90% neutral regions
                cna_data[:, i] = np.random.normal(0, 0.2, self.n_patients)
        
        # Clip to reasonable range
        cna_data = np.clip(cna_data, -3, 3)
        
        # Create DataFrame
        cna_df = pd.DataFrame(cna_data, 
                             index=self.patient_ids, 
                             columns=gene_symbols)
        cna_df.index.name = 'bcr_patient_barcode'
        
        return cna_df
    
    def _get_gene_symbols(self, n_genes):
        """Generate realistic gene symbols"""
        # Common gene prefixes in human genome
        prefixes = ['ENSG', 'TP53', 'BRCA', 'EGFR', 'KRAS', 'PIK3CA', 'PTEN', 
                   'RB1', 'CDKN2A', 'MYC', 'CCND1', 'CDK4', 'MDM2', 'VEGFA',
                   'ERBB2', 'AKT1', 'MTOR', 'FGFR', 'PDGFR', 'KIT']
        
        gene_symbols = []
        for i in range(n_genes):
            if i < 1000:  # Use some known cancer genes
                prefix = np.random.choice(prefixes)
                gene_symbols.append(f"{prefix}{i:04d}")
            else:  # Generate generic gene IDs
                gene_symbols.append(f"GENE{i:05d}")
        
        return gene_symbols
    
    def save_all_data(self, output_dir="tcga_data"):
        """Generate and save all datasets"""
        print(f"Generating complete TCGA-HNSC dataset for {self.n_patients} patients...")
        print(f"Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all datasets
        print("\n1. Generating clinical data...")
        clinical_df = self.generate_clinical_data()
        clinical_df.to_csv(os.path.join(output_dir, "clinical_data.csv"), index=False)
        print(f"Saved: clinical_data.csv ({clinical_df.shape})")
        
        print("\n2. Generating mRNA expression data...")
        mrna_df = self.generate_mrna_data()
        mrna_df.to_csv(os.path.join(output_dir, "mrna_expression.csv"))
        print(f"Saved: mrna_expression.csv ({mrna_df.shape})")
        
        print("\n3. Generating DNA methylation data...")
        methyl_df = self.generate_methylation_data()
        methyl_df.to_csv(os.path.join(output_dir, "dna_methylation.csv"))
        print(f"Saved: dna_methylation.csv ({methyl_df.shape})")
        
        print("\n4. Generating Copy Number Alteration data...")
        cna_df = self.generate_cna_data()
        cna_df.to_csv(os.path.join(output_dir, "cna_data.csv"))
        print(f"Saved: cna_data.csv ({cna_df.shape})")
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'n_patients': self.n_patients,
            'random_seed': self.random_seed,
            'clinical_features': clinical_df.shape[1],
            'mrna_genes': mrna_df.shape[1],
            'methylation_probes': methyl_df.shape[1],
            'cna_genes': cna_df.shape[1]
        }
        
        pd.Series(metadata).to_csv(os.path.join(output_dir, "metadata.csv"))
        print(f"\nDataset generation completed!")
        print(f"Total files created: 5")
        print(f"Patients: {self.n_patients}")
        print(f"Clinical features: {clinical_df.shape[1]}")
        print(f"mRNA genes: {mrna_df.shape[1]}")
        print(f"Methylation probes: {methyl_df.shape[1]}")
        print(f"CNA genes: {cna_df.shape[1]}")
        
        return clinical_df, mrna_df, methyl_df, cna_df

def load_generated_data(data_dir="tcga_data"):
    """Load the generated datasets"""
    print(f"Loading data from {data_dir}...")
    
    clinical_df = pd.read_csv(os.path.join(data_dir, "clinical_data.csv"))
    mrna_df = pd.read_csv(os.path.join(data_dir, "mrna_expression.csv"), index_col=0)
    methyl_df = pd.read_csv(os.path.join(data_dir, "dna_methylation.csv"), index_col=0)
    cna_df = pd.read_csv(os.path.join(data_dir, "cna_data.csv"), index_col=0)
    
    print("Data loaded successfully!")
    print(f"Clinical: {clinical_df.shape}")
    print(f"mRNA: {mrna_df.shape}")
    print(f"Methylation: {methyl_df.shape}")
    print(f"CNA: {cna_df.shape}")
    
    return clinical_df, mrna_df, methyl_df, cna_df

if __name__ == "__main__":
    # Generate the complete dataset
    generator = RealisticTCGAGenerator(n_patients=528, random_seed=42)
    clinical_df, mrna_df, methyl_df, cna_df = generator.save_all_data()
    
    # Display sample of clinical data
    print("\nSample of generated clinical data:")
    print(clinical_df.head())
    
    print("\nSurvival data summary:")
    print(f"Median survival time: {clinical_df['os_time'].median():.1f} days")
    print(f"Death events: {clinical_df['os_event'].sum()}/{len(clinical_df)} ({clinical_df['os_event'].mean()*100:.1f}%)")
    
    print("\nAge distribution:")
    print(clinical_df['age_at_initial_pathologic_diagnosis'].describe())
    
    print("\nStage distribution:")
    print(clinical_df['ajcc_pathologic_tumor_stage'].value_counts())
    
    print("\nDataset generation complete! Files saved in 'tcga_data' directory.")