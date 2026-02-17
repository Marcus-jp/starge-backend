# generate_enhanced_data.py
import pandas as pd
import numpy as np
import random

def main():
    print("=" * 70)
    print("ENHANCED DISEASE DATASET GENERATOR")
    print("=" * 70)
    
    # Check if dataset exists
    try:
        df = pd.read_csv('kenyan_diseases_dataset.csv')
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   Diseases: {df['disease'].nunique()}")
    except FileNotFoundError:
        print("❌ Error: 'kenyan_diseases_dataset.csv' not found!")
        print("   Make sure it's in the same folder.")
        return
    
    # Add 5 enhanced features
    enhanced_features = ['fever_pattern', 'pain_location', 'rash_type', 'stool_character', 'urine_character']
    for feature in enhanced_features:
        df[feature] = 0
    
    print(f"\n🔄 Adding enhanced features...")
    
    # Define disease patterns
    patterns = {
        'Malaria': {'fever_pattern': 2, 'pain_location': 2, 'rash_type': 0, 'stool_character': 0, 'urine_character': 1},
        'Dengue': {'fever_pattern': 3, 'pain_location': 1, 'rash_type': 2, 'stool_character': 0, 'urine_character': 0},
        'Measles': {'fever_pattern': 1, 'pain_location': 5, 'rash_type': 2, 'stool_character': 0, 'urine_character': 0},
        'Typhoid': {'fever_pattern': 1, 'pain_location': 3, 'rash_type': 3, 'stool_character': 1, 'urine_character': 0},
        'Cholera': {'fever_pattern': 0, 'pain_location': 3, 'rash_type': 0, 'stool_character': 2, 'urine_character': 0},
        'Gastroenteritis': {'fever_pattern': 1, 'pain_location': 3, 'rash_type': 0, 'stool_character': 1, 'urine_character': 0},
        'Hepatitis': {'fever_pattern': 1, 'pain_location': 3, 'rash_type': 0, 'stool_character': 3, 'urine_character': 1},
        'Diabetes': {'fever_pattern': 0, 'pain_location': 0, 'rash_type': 0, 'stool_character': 0, 'urine_character': 3},
        'UTI': {'fever_pattern': 1, 'pain_location': 3, 'rash_type': 0, 'stool_character': 0, 'urine_character': 4},
        'COVID-19': {'fever_pattern': 1, 'pain_location': 2, 'rash_type': 0, 'stool_character': 0, 'urine_character': 0},
        'Pneumonia': {'fever_pattern': 1, 'pain_location': 4, 'rash_type': 0, 'stool_character': 0, 'urine_character': 0},
        'Tuberculosis': {'fever_pattern': 1, 'pain_location': 4, 'rash_type': 0, 'stool_character': 0, 'urine_character': 0},
        'Asthma': {'fever_pattern': 0, 'pain_location': 4, 'rash_type': 0, 'stool_character': 0, 'urine_character': 0},
        'Hypertension': {'fever_pattern': 0, 'pain_location': 5, 'rash_type': 0, 'stool_character': 0, 'urine_character': 0},
        'HIV_AIDS': {'fever_pattern': 1, 'pain_location': 0, 'rash_type': 1, 'stool_character': 1, 'urine_character': 0},
        'Chickenpox': {'fever_pattern': 1, 'pain_location': 0, 'rash_type': 1, 'stool_character': 0, 'urine_character': 0},
        'Bronchitis': {'fever_pattern': 1, 'pain_location': 4, 'rash_type': 0, 'stool_character': 0, 'urine_character': 0},
        'Rheumatic_Fever': {'fever_pattern': 1, 'pain_location': 1, 'rash_type': 2, 'stool_character': 0, 'urine_character': 0},
        'Brucellosis': {'fever_pattern': 2, 'pain_location': 1, 'rash_type': 0, 'stool_character': 0, 'urine_character': 0},
        'Leptospirosis': {'fever_pattern': 1, 'pain_location': 2, 'rash_type': 0, 'stool_character': 0, 'urine_character': 1}
    }
    
    # Apply patterns with some randomness
    for idx, row in df.iterrows():
        disease = row['disease']
        if disease in patterns:
            for feature, value in patterns[disease].items():
                # Add slight variation for realism
                if value > 0:
                    variation = random.choice([-1, 0, 0, 1])  # Mostly keep same
                    final_value = max(0, value + variation)
                else:
                    final_value = 0
                df.at[idx, feature] = final_value
    
    # Save enhanced dataset
    output_file = 'kenyan_diseases_enhanced.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved as: {output_file}")
    print(f"   • Total features: {df.shape[1]}")
    print(f"   • Enhanced features added: {', '.join(enhanced_features)}")
    
    # Show sample
    print("\n🔍 Sample of enhanced data:")
    sample_diseases = ['Malaria', 'Dengue', 'Typhoid', 'Diabetes']
    for disease in sample_diseases:
        if disease in df['disease'].values:
            sample = df[df['disease'] == disease].iloc[0]
            print(f"\n{disease}:")
            for feature in enhanced_features:
                value = sample[feature]
                print(f"  {feature}: {value}")
    
    print("\n✅ Enhanced dataset created successfully!")
    print("\n🚀 Next steps:")
    print("1. Train enhanced model: python train_starge.py")
    print("2. Update predict.py to use enhanced features")

if __name__ == "__main__":
    main()