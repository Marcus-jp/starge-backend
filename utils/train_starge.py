# train_starge.py - UPDATED VERSION
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse
import warnings
import os
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train STARGE AI Disease Prediction Model')
    parser.add_argument('--enhanced', action='store_true', 
                       help='Use enhanced dataset with 5 additional features')
    parser.add_argument('--basic', action='store_true',
                       help='Use basic dataset (40 symptoms only)')
    parser.add_argument('--compare', action='store_true',
                       help='Train both models and compare results')
    parser.add_argument('--output', type=str, default='starge_model',
                       help='Base name for output files')
    return parser.parse_args()

# ========== MAIN TRAINING FUNCTION ==========
def train_model(data_file, model_name, is_enhanced=False):
    """Train a model on the given dataset"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    
    # Load dataset
    print(f"\n📂 Loading dataset: {data_file}")
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Successfully loaded")
        print(f"   • Rows: {df.shape[0]}")
        print(f"   • Features: {df.shape[1] - 1}")
        print(f"   • Diseases: {df['disease'].nunique()}")
    except FileNotFoundError:
        print(f"❌ Error: {data_file} not found!")
        if is_enhanced:
            print("   Run: python generate_enhanced_data.py first")
        return None
    
    # Check if enhanced features are present
    enhanced_features = ['fever_pattern', 'pain_location', 'rash_type', 'stool_character', 'urine_character']
    has_enhanced = all(feature in df.columns for feature in enhanced_features)
    
    if is_enhanced and not has_enhanced:
        print("⚠️  Warning: Dataset doesn't have enhanced features")
        print("   Run: python generate_enhanced_data.py first")
        return None
    
    # Prepare data
    print("\n🔄 Preparing data...")
    X = df.drop('disease', axis=1)
    y = df['disease']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"📋 Diseases ({len(label_encoder.classes_)}):")
    diseases_per_line = 4
    for i in range(0, len(label_encoder.classes_), diseases_per_line):
        line = []
        for j in range(diseases_per_line):
            if i + j < len(label_encoder.classes_):
                disease = label_encoder.classes_[i + j]
                line.append(f"{i+j:2d}: {disease}")
        print("   " + " | ".join(line))
    
    # Split data
    print("\n✂️ Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=y_encoded
    )
    
    print(f"   • Training samples: {X_train.shape[0]}")
    print(f"   • Testing samples:  {X_test.shape[0]}")
    
    # Train model
    print("\n🤖 Training Random Forest Classifier...")
    
    if is_enhanced:
        # Use more complex model for enhanced features
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True
        )
    else:
        # Simpler model for basic features
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")
    
    # Evaluate
    print("\n📊 Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Cross-validation
    print("\n   Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X, y_encoded, cv=5, n_jobs=-1)
    print(f"   • CV Scores: {cv_scores.round(4)}")
    print(f"   • Mean CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    if hasattr(model, 'oob_score_'):
        print(f"   • Out-of-bag score: {model.oob_score_:.4f}")
    
    # Feature importance
    print("\n📊 Feature importance analysis...")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 15 Most Important Features:")
    for i, row in enumerate(feature_importance.head(15).itertuples(), 1):
        marker = "🌟" if is_enhanced and row.feature in enhanced_features else "  "
        print(f"   {i:2d}. {marker} {row.feature:25s} - {row.importance:.4f}")
    
    # Save model
    print("\n💾 Saving model files...")
    
    # Create filenames
    model_filename = f"{model_name}.pkl"
    encoder_filename = f"{model_name}_encoder.pkl"
    features_filename = f"{model_name}_features.pkl"
    
    joblib.dump(model, model_filename)
    joblib.dump(label_encoder, encoder_filename)
    joblib.dump(list(X.columns), features_filename)
    
    print(f"✅ Model saved: {model_filename}")
    print(f"✅ Encoder saved: {encoder_filename}")
    print(f"✅ Features saved: {features_filename}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'dataset': data_file,
        'is_enhanced': is_enhanced,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_shape': df.shape,
        'num_diseases': len(label_encoder.classes_),
        'num_features': X.shape[1],
        'accuracy': accuracy,
        'cv_mean_score': cv_scores.mean(),
        'feature_names': list(X.columns),
        'diseases': list(label_encoder.classes_)
    }
    
    metadata_filename = f"{model_name}_metadata.pkl"
    joblib.dump(metadata, metadata_filename)
    print(f"✅ Metadata saved: {metadata_filename}")
    
    return {
        'model': model,
        'encoder': label_encoder,
        'features': X.columns,
        'accuracy': accuracy,
        'cv_score': cv_scores.mean(),
        'is_enhanced': is_enhanced
    }

# ========== COMPARE MODELS ==========
def compare_models(basic_result, enhanced_result):
    """Compare basic and enhanced models"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON: BASIC vs ENHANCED")
    print(f"{'='*60}")
    
    print(f"\n📊 Performance Comparison:")
    print(f"{'Metric':<20} {'Basic':<15} {'Enhanced':<15} {'Improvement':<15}")
    print(f"{'-'*60}")
    
    basic_acc = basic_result['accuracy']
    enhanced_acc = enhanced_result['accuracy']
    improvement = enhanced_acc - basic_acc
    
    print(f"{'Accuracy':<20} {basic_acc:.4f} ({basic_acc*100:.2f}%)  "
          f"{enhanced_acc:.4f} ({enhanced_acc*100:.2f}%)  "
          f"{improvement*100:+.2f}%")
    
    basic_cv = basic_result['cv_score']
    enhanced_cv = enhanced_result['cv_score']
    cv_improvement = enhanced_cv - basic_cv
    
    print(f"{'CV Score':<20} {basic_cv:.4f} ({basic_cv*100:.2f}%)  "
          f"{enhanced_cv:.4f} ({enhanced_cv*100:.2f}%)  "
          f"{cv_improvement*100:+.2f}%")
    
    print(f"\n🎯 Enhanced model has {improvement*100:+.1f}% better accuracy")
    
    if improvement > 0:
        print("✅ Enhanced features are helping!")
    else:
        print("⚠️  Basic model performed better (check enhanced data quality)")
    
    # Test differentiation on similar diseases
    print(f"\n🔍 Testing disease differentiation...")
    
    test_cases = [
        ("Malaria-like", ['fever', 'chills', 'headache', 'body_aches']),
        ("Typhoid-like", ['fever', 'abdominal_pain', 'diarrhea', 'headache']),
        ("Diabetes-like", ['fatigue', 'frequent_urination', 'excessive_thirst'])
    ]
    
    for case_name, symptoms in test_cases:
        print(f"\n   {case_name} symptoms:")
        
        for model_name, result in [('Basic', basic_result), ('Enhanced', enhanced_result)]:
            # Create test input
            test_input = [0] * len(result['features'])
            for symptom in symptoms:
                if symptom in result['features']:
                    idx = list(result['features']).index(symptom)
                    test_input[idx] = 1
            
            # Add enhanced features if available
            if result['is_enhanced']:
                enhanced_features = ['fever_pattern', 'pain_location', 'rash_type', 'stool_character', 'urine_character']
                for feature in enhanced_features:
                    if feature in result['features']:
                        idx = list(result['features']).index(feature)
                        # Set appropriate values based on case
                        if case_name == 'Malaria-like':
                            if feature == 'fever_pattern':
                                test_input[idx] = 2  # Cyclical
                            elif feature == 'pain_location':
                                test_input[idx] = 2  # Muscle
                        elif case_name == 'Typhoid-like':
                            if feature == 'fever_pattern':
                                test_input[idx] = 1  # Continuous
                            elif feature == 'pain_location':
                                test_input[idx] = 3  # Abdominal
            
            # Predict
            test_df = pd.DataFrame([test_input], columns=result['features'])
            prediction = result['model'].predict(test_df)[0]
            probabilities = result['model'].predict_proba(test_df)[0]
            
            disease = result['encoder'].inverse_transform([prediction])[0]
            confidence = probabilities[prediction] * 100
            
            # Get top 3
            top_3_idx = np.argsort(probabilities)[::-1][:3]
            top_diseases = result['encoder'].inverse_transform(top_3_idx)
            top_confidences = probabilities[top_3_idx] * 100
            
            print(f"     • {model_name}: {disease} ({confidence:.1f}%)")
            print(f"       Top 3: ", end="")
            for d, c in zip(top_diseases, top_confidences):
                print(f"{d}({c:.1f}%) ", end="")
            print()

# ========== MAIN ==========
def main():
    args = parse_args()
    
    print("=" * 70)
    print("STARGE AI - DISEASE PREDICTION MODEL TRAINER")
    print("=" * 70)
    
    # Determine what to train
    if args.compare or (not args.basic and not args.enhanced):
        # Train both if compare flag or no flags specified
        train_both = True
    else:
        train_both = False
    
    results = {}
    
    # Train basic model
    if train_both or args.basic:
        basic_result = train_model(
            data_file="kenyan_diseases_dataset.csv",
            model_name=f"{args.output}_basic",
            is_enhanced=False
        )
        if basic_result:
            results['basic'] = basic_result
    
    # Train enhanced model
    if train_both or args.enhanced:
        enhanced_result = train_model(
            data_file="kenyan_diseases_enhanced.csv",
            model_name=f"{args.output}_enhanced",
            is_enhanced=True
        )
        if enhanced_result:
            results['enhanced'] = enhanced_result
    
    # Compare if both were trained
    if 'basic' in results and 'enhanced' in results:
        compare_models(results['basic'], results['enhanced'])
    
    # Final instructions
    print(f"\n{'='*70}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\n📁 FILES CREATED:")
    if 'basic' in results:
        print(f"   Basic Model:")
        print(f"     • {args.output}_basic.pkl")
        print(f"     • {args.output}_basic_encoder.pkl")
        print(f"     • {args.output}_basic_features.pkl")
        print(f"     • {args.output}_basic_metadata.pkl")
    
    if 'enhanced' in results:
        print(f"   Enhanced Model:")
        print(f"     • {args.output}_enhanced.pkl")
        print(f"     • {args.output}_enhanced_encoder.pkl")
        print(f"     • {args.output}_enhanced_features.pkl")
        print(f"     • {args.output}_enhanced_metadata.pkl")
    
    print(f"\n🚀 HOW TO USE:")
    print(f"   1. For prediction GUI: Update predict.py to load correct model")
    print(f"   2. Basic model: Use when you only have 40 binary symptoms")
    print(f"   3. Enhanced model: Use when you have additional feature info")
    
    print(f"\n💡 TIPS:")
    print(f"   • Enhanced model needs 5 extra features in predict.py")
    print(f"   • Enhanced features: fever_pattern, pain_location, rash_type,")
    print(f"     stool_character, urine_character")
    
    print(f"\n📊 Expected accuracy improvements:")
    print(f"   • Basic model: 80-85%")
    print(f"   • Enhanced model: 85-92% (better at similar diseases)")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()