import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, '..', 'datasets')
UTILS_DIR = BASE_DIR

# ==================== TRAIN BASIC MODEL ====================
print("="*50)
print("TRAINING BASIC MODEL")
print("="*50)

# Load basic dataset
basic_csv = os.path.join(DATASETS_DIR, 'kenyan_diseases_dataset.csv')
print(f"Loading basic dataset from: {basic_csv}")

df_basic = pd.read_csv(basic_csv)
print(f"Basic dataset shape: {df_basic.shape}")
print(f"Columns: {df_basic.columns.tolist()}")

# Identify feature columns (everything except 'disease' and enhanced features)
enhanced_cols = ['fever_pattern', 'pain_location', 'rash_type', 'stool_character', 'urine_character']
feature_cols_basic = [col for col in df_basic.columns if col != 'disease' and col not in enhanced_cols]

print(f"\nBasic features ({len(feature_cols_basic)}): {feature_cols_basic}")

X_basic = df_basic[feature_cols_basic]
y_basic = df_basic['disease']

print(f"X_basic shape: {X_basic.shape}")
print(f"y_basic value counts:\n{y_basic.value_counts()}")

# Train-test split
X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(
    X_basic, y_basic, test_size=0.2, random_state=42, stratify=y_basic
)

# Train model
print("\nTraining RandomForestClassifier (basic)...")
model_basic = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model_basic.fit(X_train_basic, y_train_basic)

# Evaluate
train_score_basic = model_basic.score(X_train_basic, y_train_basic)
test_score_basic = model_basic.score(X_test_basic, y_test_basic)
print(f"Basic Model - Train Score: {train_score_basic:.4f}")
print(f"Basic Model - Test Score: {test_score_basic:.4f}")

# Encode labels
le_basic = LabelEncoder()
le_basic.fit(y_basic)

# Save basic model
model_path_basic = os.path.join(UTILS_DIR, 'starge_model_basic.pkl')
encoder_path_basic = os.path.join(UTILS_DIR, 'starge_model_basic_encoder.pkl')
features_path_basic = os.path.join(UTILS_DIR, 'starge_model_basic_features.pkl')

joblib.dump(model_basic, model_path_basic)
joblib.dump(le_basic, encoder_path_basic)
joblib.dump(feature_cols_basic, features_path_basic)

print(f"✓ Basic model saved to: {model_path_basic}")
print(f"✓ Basic encoder saved to: {encoder_path_basic}")
print(f"✓ Basic features saved to: {features_path_basic}")

# ==================== TRAIN ENHANCED MODEL ====================
print("\n" + "="*50)
print("TRAINING ENHANCED MODEL")
print("="*50)

# Load enhanced dataset
enhanced_csv = os.path.join(DATASETS_DIR, 'kenyan_diseases_enhanced.csv')
print(f"Loading enhanced dataset from: {enhanced_csv}")

df_enhanced = pd.read_csv(enhanced_csv)
print(f"Enhanced dataset shape: {df_enhanced.shape}")
print(f"Columns: {df_enhanced.columns.tolist()}")

# Get all feature columns (including enhanced ones)
feature_cols_enhanced = [col for col in df_enhanced.columns if col != 'disease']

print(f"\nEnhanced features ({len(feature_cols_enhanced)}): {feature_cols_enhanced}")

X_enhanced = df_enhanced[feature_cols_enhanced]
y_enhanced = df_enhanced['disease']

print(f"X_enhanced shape: {X_enhanced.shape}")
print(f"y_enhanced value counts:\n{y_enhanced.value_counts()}")

# Train-test split
X_train_enhanced, X_test_enhanced, y_train_enhanced, y_test_enhanced = train_test_split(
    X_enhanced, y_enhanced, test_size=0.2, random_state=42, stratify=y_enhanced
)

# Train model
print("\nTraining RandomForestClassifier (enhanced)...")
model_enhanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model_enhanced.fit(X_train_enhanced, y_train_enhanced)

# Evaluate
train_score_enhanced = model_enhanced.score(X_train_enhanced, y_train_enhanced)
test_score_enhanced = model_enhanced.score(X_test_enhanced, y_test_enhanced)
print(f"Enhanced Model - Train Score: {train_score_enhanced:.4f}")
print(f"Enhanced Model - Test Score: {test_score_enhanced:.4f}")

# Encode labels
le_enhanced = LabelEncoder()
le_enhanced.fit(y_enhanced)

# Save enhanced model
model_path_enhanced = os.path.join(UTILS_DIR, 'starge_model_enhanced.pkl')
encoder_path_enhanced = os.path.join(UTILS_DIR, 'starge_model_enhanced_encoder.pkl')
features_path_enhanced = os.path.join(UTILS_DIR, 'starge_model_enhanced_features.pkl')

joblib.dump(model_enhanced, model_path_enhanced)
joblib.dump(le_enhanced, encoder_path_enhanced)
joblib.dump(feature_cols_enhanced, features_path_enhanced)

print(f"✓ Enhanced model saved to: {model_path_enhanced}")
print(f"✓ Enhanced encoder saved to: {encoder_path_enhanced}")
print(f"✓ Enhanced features saved to: {features_path_enhanced}")

# ==================== SUMMARY ====================
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"\nBasic Model:")
print(f"  - Test Accuracy: {test_score_basic:.2%}")
print(f"  - Features: {len(feature_cols_basic)}")

print(f"\nEnhanced Model:")
print(f"  - Test Accuracy: {test_score_enhanced:.2%}")
print(f"  - Features: {len(feature_cols_enhanced)}")

print(f"\nAll models saved to: {UTILS_DIR}")