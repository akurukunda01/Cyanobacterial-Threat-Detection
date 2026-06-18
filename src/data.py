import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_and_process_data():
    """
    Load and process the NLA 2017 phytoplankton, profile, and water chemistry data
    to create a dataset for predicting cyanobacteria percentage.
    """
    # ============================================
    # STEP 1: Load the phytoplankton data
    # ============================================
    phyto = pd.read_csv(os.path.join(DATA_DIR, 'nla_2017_phytoplankton_count-data.csv'), encoding='latin1')

    # Check what columns you have
    print(phyto.columns)
    print(phyto.head())

    # ============================================
    # STEP 2: Calculate cyanobacteria %
    # ============================================

    # Filter for cyanobacteria (BLUE-GREEN ALGAE)
    cyano = phyto[phyto['ALGAL_GROUP'] == 'BLUE-GREEN ALGAE']

    # Sum cyanobacteria biovolume by lake (UID)
    cyano_biovolume = cyano.groupby('UID')['BIOVOLUME'].sum().reset_index()
    cyano_biovolume.columns = ['UID', 'CYANO_BIOVOLUME']

    # Sum total phytoplankton biovolume by lake
    total_biovolume = phyto.groupby('UID')['BIOVOLUME'].sum().reset_index()
    total_biovolume.columns = ['UID', 'TOTAL_BIOVOLUME']

    # Merge and calculate percentage
    cyano_pct = cyano_biovolume.merge(total_biovolume, on='UID', how='right')
    cyano_pct['CYANO_BIOVOLUME'] = cyano_pct['CYANO_BIOVOLUME'].fillna(0)  # Lakes with no cyano
    cyano_pct['CYANO_PCT'] = (cyano_pct['CYANO_BIOVOLUME'] / cyano_pct['TOTAL_BIOVOLUME']) * 100

    print(f"\nCyano % calculated for {len(cyano_pct)} lakes")
    print(cyano_pct[['UID', 'CYANO_PCT']].head())

    # ============================================
    # STEP 3: Load profile data (4 sensors)
    # ============================================
    profile = pd.read_csv(os.path.join(DATA_DIR, 'nla_2017_profile-data.csv'), encoding='latin1')

    # Check columns
    print("\nProfile columns:")
    print(profile.columns)

    # Filter for SURFACE measurements only (DEPTH = 0 or minimum depth)
    surface = profile[profile['DEPTH'] <= 1.0]  # Adjust threshold if needed

    # Convert sensor columns to numeric (handles text like "NA", "NR", etc.)
    print("\nConverting sensor columns to numeric...")
    surface['TEMPERATURE'] = pd.to_numeric(surface['TEMPERATURE'], errors='coerce')
    surface['PH'] = pd.to_numeric(surface['PH'], errors='coerce')
    surface['OXYGEN'] = pd.to_numeric(surface['OXYGEN'], errors='coerce')
    surface['CONDUCTIVITY'] = pd.to_numeric(surface['CONDUCTIVITY'], errors='coerce')
    
    # Check data types
    print("\nData types after conversion:")
    print(surface[['TEMPERATURE', 'PH', 'OXYGEN', 'CONDUCTIVITY']].dtypes)
    
    # Check for any remaining issues
    print("\nChecking for non-numeric values:")
    print(f"Temperature nulls: {surface['TEMPERATURE'].isna().sum()}")
    print(f"PH nulls: {surface['PH'].isna().sum()}")
    print(f"Oxygen nulls: {surface['OXYGEN'].isna().sum()}")
    print(f"Conductivity nulls: {surface['CONDUCTIVITY'].isna().sum()}")

    # Get the 4 sensor parameters for each lake
    sensors = surface.groupby('UID').agg({
        'TEMPERATURE': 'mean',    # Average surface temp
        'PH': 'mean',             # Average surface pH
        'OXYGEN': 'mean',         # Average surface DO
        'CONDUCTIVITY': 'mean'    # Average surface conductivity
    }).reset_index()

    print(f"\nSensor data for {len(sensors)} lakes")
    print(sensors.head())

    # ============================================
    # STEP 4: Load water chemistry data (TURBIDITY)
    # ============================================
    chem = pd.read_csv(os.path.join(DATA_DIR, 'nla_2017_water_chemistry_chla-data.csv'), encoding='latin1')
    
    print("\nWater chemistry columns:")
    print(chem.columns)
    
    # Filter for turbidity measurements
    turb_data = chem[chem['ANALYTE'] == 'TURB'].copy()
    
    # Convert RESULT to numeric
    turb_data['RESULT'] = pd.to_numeric(turb_data['RESULT'], errors='coerce')
    
    # Get turbidity for each UID (average if multiple measurements)
    turbidity = turb_data.groupby('UID')['RESULT'].mean().reset_index()
    turbidity.columns = ['UID', 'TURB']
    
    print(f"\nTurbidity data for {len(turbidity)} lakes")
    print(turbidity.head())

    # ============================================
    # STEP 5: Merge everything together
    # ============================================
    # First merge cyano % with sensors
    final_data = cyano_pct[['UID', 'CYANO_PCT']].merge(sensors, on='UID', how='inner')
    
    # Then add turbidity
    final_data = final_data.merge(turbidity, on='UID', how='inner')

    # Remove rows with missing values
    print(f"\nBefore removing nulls: {len(final_data)} lakes")
    final_data = final_data.dropna()
    print(f"After removing nulls: {len(final_data)} lakes")

    print(f"\n✅ Final dataset: {len(final_data)} lakes with complete data")
    print("\nColumns:", final_data.columns.tolist())
    print(final_data.head())
    print("\n" + "="*60)
    print("Dataset Summary:")
    print("="*60)
    print(final_data.describe())

    # Save your clean dataset
    out_path = os.path.join(DATA_DIR, 'cyano_prediction_data_with_turb.csv')
    final_data.to_csv(out_path, index=False)
    print(f"\n✅ Saved to: {out_path}")
    



    
    


def train_model():
    """
    Placeholder function for training a model to predict cyanobacteria percentage.
    """ 
    data = pd.read_csv(os.path.join(DATA_DIR, 'cyano_prediction_data_with_turb.csv'))
    X = data[['TEMPERATURE', 'PH', 'OXYGEN', 'CONDUCTIVITY']]
    y = data['CYANO_PCT']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")

    # Store results
    results = []

    # ============================================
    # 1. Linear Regression
    # ============================================
    print("Training Linear Regression...")
    start = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    train_time = time.time()-start

    y_pred = lr_model.predict(X_test)
    results.append({
        'Model': 'Linear Regression',
        'R²': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Training_Time': train_time
    })
    print(f"Done in {train_time:.2f} seconds\n")

    # ============================================
    # 2. Random Forest
    # ============================================
    print("Training Random Forest...")
    start = time.time()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    train_time = time.time()-start

    y_pred = rf_model.predict(X_test)
    results.append({
        'Model': 'Random Forest',
        'R²': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Training_Time': train_time
    })
    print(f"Done in {train_time:.2f} seconds\n")

    # 3. Gradient Boosting (sklearn's version - no XGBoost needed!)
    # ============================================
    print("Training Gradient Boosting...")
    start = time.time()
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    train_time = time.time()-start

    y_pred = gb_model.predict(X_test)
    results.append({
        'Model': 'Gradient Boosting',
        'R²': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Training_Time': train_time
    })
    print(f"Done in {train_time:.2f} seconds\n")

    # ============================================
    # 4. Neural Network
    # ============================================
    print("Training Neural Network...")
    start = time.time()
    nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    nn_model.fit(X_train, y_train)
    train_time = time.time()-start

    y_pred = nn_model.predict(X_test)
    results.append({
        'Model': 'Neural Network',
        'R²': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Training_Time': train_time
    })
    print(f"Done in {train_time:.2f} seconds\n")

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*70)
    print("📊 SUMMARY - MODEL COMPARISON")
    print("="*70 + "\n")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R²', ascending=False)

    # Print formatted table
    print(f"{'Model':<20} {'R²':<10} {'MAE (%)':<12} {'RMSE (%)':<12} {'Time (s)':<10}")
    print("-"*70)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<20} {row['R²']:<10.4f} {row['MAE']:<12.2f} {row['RMSE']:<12.2f} {row['Training_Time']:<10.2f}")

    print("\n" + "="*70)
    print("🏆 BEST MODEL")
    print("="*70)
    best = results_df.iloc[0]
    print(f"Model: {best['Model']}")
    print(f"R² Score: {best['R²']:.4f} (explains {best['R²']*100:.1f}% of variance)")
    print(f"MAE: {best['MAE']:.2f}% (average error)")
    print(f"RMSE: {best['RMSE']:.2f}%")
    print(f"Training Time: {best['Training_Time']:.2f} seconds")

    print("\n" + "="*70)
    print("💡 INTERPRETATION")
    print("="*70)
    print(f"With just 4 sensors (Temp, pH, DO, Conductivity), the {best['Model']}")
    print(f"can predict cyanobacteria abundance with ~{best['MAE']:.1f}% average error.")
    print(f"This model is ready for deployment on your ROV! 🤖")
    print("="*70 + "\n")

def explore_water_chemistry():
    """
    Explore the NLA 2017 water chemistry data to see available analytes.
    """

    # Load water chemistry
    chem = pd.read_csv(os.path.join(DATA_DIR, 'nla_2017_water_chemistry_chla-data.csv'), encoding='latin-1')

    # See what analytes are available
    print("Available analytes:")
    print(chem['ANALYTE'].unique())


def load_and_process_data_classification(threshold=20):
    """
    Load and process the NLA 2017 data for CLASSIFICATION task.
    Predicts: Bloom Present (YES/NO) instead of exact percentage.
    
    Parameters:
    -----------
    threshold : float, default=20
        Cyanobacteria percentage threshold to define a bloom.
        Values > threshold = Bloom (1), Values <= threshold = No Bloom (0)
    
    Returns:
    --------
    pandas.DataFrame with binary classification target
    """
    
    print("="*70)
    print("LOADING DATA FOR CLASSIFICATION (Bloom Presence Detection)")
    print("="*70)
    
    # ============================================
    # STEP 1: Load the phytoplankton data
    # ============================================
    print("\n📂 Loading phytoplankton data...")
    phyto = pd.read_csv(os.path.join(DATA_DIR, 'nla_2017_phytoplankton_count-data.csv'), encoding='latin1')
    print(f"   Loaded {len(phyto)} phytoplankton records")

    # ============================================
    # STEP 2: Calculate cyanobacteria %
    # ============================================
    print("\n🦠 Calculating cyanobacteria percentage...")
    
    # Filter for cyanobacteria (BLUE-GREEN ALGAE)
    cyano = phyto[phyto['ALGAL_GROUP'] == 'BLUE-GREEN ALGAE']

    # Sum cyanobacteria biovolume by lake (UID)
    cyano_biovolume = cyano.groupby('UID')['BIOVOLUME'].sum().reset_index()
    cyano_biovolume.columns = ['UID', 'CYANO_BIOVOLUME']

    # Sum total phytoplankton biovolume by lake
    total_biovolume = phyto.groupby('UID')['BIOVOLUME'].sum().reset_index()
    total_biovolume.columns = ['UID', 'TOTAL_BIOVOLUME']

    # Merge and calculate percentage
    cyano_pct = cyano_biovolume.merge(total_biovolume, on='UID', how='right')
    cyano_pct['CYANO_BIOVOLUME'] = cyano_pct['CYANO_BIOVOLUME'].fillna(0)
    cyano_pct['CYANO_PCT'] = (cyano_pct['CYANO_BIOVOLUME'] / cyano_pct['TOTAL_BIOVOLUME']) * 100
    
    print(f"   Calculated cyano % for {len(cyano_pct)} lakes")
    print(f"   Range: {cyano_pct['CYANO_PCT'].min():.2f}% - {cyano_pct['CYANO_PCT'].max():.2f}%")
    print(f"   Mean: {cyano_pct['CYANO_PCT'].mean():.2f}%")

    # ============================================
    # STEP 3: Load profile data (4 sensors)
    # ============================================
    print("\n🌡️  Loading sensor data...")
    profile = pd.read_csv(os.path.join(DATA_DIR, 'nla_2017_profile-data.csv'), encoding='latin1')

    # Filter for SURFACE measurements only
    surface = profile[profile['DEPTH'] <= 1.0]

    # Convert sensor columns to numeric
    surface['TEMPERATURE'] = pd.to_numeric(surface['TEMPERATURE'], errors='coerce')
    surface['PH'] = pd.to_numeric(surface['PH'], errors='coerce')
    surface['OXYGEN'] = pd.to_numeric(surface['OXYGEN'], errors='coerce')
    surface['CONDUCTIVITY'] = pd.to_numeric(surface['CONDUCTIVITY'], errors='coerce')

    # Get the 4 sensor parameters for each lake
    sensors = surface.groupby('UID').agg({
        'TEMPERATURE': 'mean',
        'PH': 'mean',
        'OXYGEN': 'mean',
        'CONDUCTIVITY': 'mean'
    }).reset_index()

    print(f"   Sensor data for {len(sensors)} lakes")

    # ============================================
    # STEP 4: Merge and create classification target
    # ============================================
    print("\n🔗 Merging datasets...")
    final_data = cyano_pct[['UID', 'CYANO_PCT']].merge(sensors, on='UID', how='inner')

    # Remove rows with missing values
    print(f"   Before removing nulls: {len(final_data)} lakes")
    final_data = final_data.dropna()
    print(f"   After removing nulls: {len(final_data)} lakes")

    # ============================================
    # STEP 5: CREATE BINARY CLASSIFICATION TARGET
    # ============================================
    print(f"\n🎯 Creating classification target (threshold = {threshold}%)...")
    
    final_data['CYANO_BLOOM'] = (final_data['CYANO_PCT'] > threshold).astype(int)
    
    # Statistics
    no_bloom = (final_data['CYANO_BLOOM'] == 0).sum()
    bloom = (final_data['CYANO_BLOOM'] == 1).sum()
    
    print(f"\n   Class Distribution:")
    print(f"   ├─ No Bloom (≤{threshold}%):  {no_bloom} lakes ({no_bloom/len(final_data)*100:.1f}%)")
    print(f"   └─ Bloom (>{threshold}%):     {bloom} lakes ({bloom/len(final_data)*100:.1f}%)")
    
    # Check class balance
    balance_ratio = min(no_bloom, bloom) / max(no_bloom, bloom)
    if balance_ratio < 0.3:
        print(f"\n   ⚠️  WARNING: Classes are imbalanced (ratio = {balance_ratio:.2f})")
        print(f"   Consider adjusting threshold or using class weights in models")
    else:
        print(f"\n   ✅ Classes are reasonably balanced (ratio = {balance_ratio:.2f})")

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*70)
    print("📊 FINAL DATASET SUMMARY")
    print("="*70)
    print(f"Total lakes: {len(final_data)}")
    print(f"\nFeatures (Input):")
    print(f"  1. TEMPERATURE (°C)")
    print(f"  2. PH")
    print(f"  3. OXYGEN (mg/L)")
    print(f"  4. CONDUCTIVITY (μS/cm)")
    print(f"\nTarget (Output):")
    print(f"  CYANO_BLOOM: 0 = No Bloom, 1 = Bloom")
    print(f"\nCyano % kept for reference (not used in training)")
    
    print("\n" + "="*70)
    print("DATASET PREVIEW")
    print("="*70)
    print(final_data.head(10))
    
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    print(final_data.describe())

    # ============================================
    # SAVE DATA
    # ============================================
    filename = os.path.join(DATA_DIR, f'cyano_classification_data_threshold_{threshold}.csv')
    final_data.to_csv(filename, index=False)
    print(f"\n✅ Saved to: {filename}")

def train_classification_model():
    """
    Train and evaluate classification models to predict cyanobacteria bloom presence.
    """

    print("="*70)
    print("TRAINING CLASSIFICATION MODELS FOR CYANOBACTERIA BLOOM DETECTION")
    print("="*70)
    
    data = pd.read_csv(os.path.join(DATA_DIR, 'cyano_classification_data_threshold_20.csv'))

    # ============================================
    # CREATE BINARY TARGET (BLOOM vs NO BLOOM)
    # ============================================
    # Define threshold: >20% cyano = bloom
    threshold = 20
    data['CYANO_BLOOM'] = (data['CYANO_PCT'] > threshold).astype(int)

    print("="*60)
    print(f"CLASSIFICATION: Bloom Threshold = {threshold}%")
    print("="*60)
    print(f"Total lakes: {len(data)}")
    print(f"No bloom (≤{threshold}%): {(data['CYANO_BLOOM']==0).sum()} ({(data['CYANO_BLOOM']==0).sum()/len(data)*100:.1f}%)")
    print(f"Bloom (>{threshold}%): {(data['CYANO_BLOOM']==1).sum()} ({(data['CYANO_BLOOM']==1).sum()/len(data)*100:.1f}%)")

    # Prepare data
    X = data[['TEMPERATURE', 'PH', 'OXYGEN', 'CONDUCTIVITY']]
    y = data['CYANO_BLOOM']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining set: {len(X_train)} lakes")
    print(f"Test set: {len(X_test)} lakes")

    # Store results
    results = []

    # ============================================
    # 1. Logistic Regression
    # ============================================
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    start = time.time()
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    train_time = time.time()-start

    y_pred = lr_model.predict(X_test)
    results.append({
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Training_Time': train_time
    })
    print(f"Done in {train_time:.2f} seconds")

    # ============================================
    # 2. Random Forest Classifier
    # ============================================
    print("\n" + "="*60)
    print("Training Random Forest Classifier...")
    start = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    train_time = time.time()-start

    y_pred = rf_model.predict(X_test)
    results.append({
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Training_Time': train_time
    })
    print(f"Done in {train_time:.2f} seconds")

    # ============================================
    # 3. Gradient Boosting Classifier
    # ============================================
    print("\n" + "="*60)
    print("Training Gradient Boosting Classifier...")
    start = time.time()
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    train_time = time.time()-start

    y_pred = gb_model.predict(X_test)
    results.append({
        'Model': 'Gradient Boosting',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Training_Time': train_time
    })
    print(f"Done in {train_time:.2f} seconds")

    # ============================================
    # 4. Neural Network Classifier
    # ============================================
    print("\n" + "="*60)
    print("Training Neural Network Classifier...")
    start = time.time()
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500)
    nn_model.fit(X_train, y_train)
    train_time = time.time()-start

    y_pred = nn_model.predict(X_test)
    results.append({
        'Model': 'Neural Network',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Training_Time': train_time
    })
    print(f"Done in {train_time:.2f} seconds")

    # ============================================
    # SUMMARY
    # ============================================
    print("\n\n" + "="*80)
    print("📊 SUMMARY - CLASSIFICATION MODEL COMPARISON")
    print("="*80 + "\n")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)

    # Print formatted table
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Time (s)':<10}")
    print("-"*90)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<25} {row['Accuracy']:<12.4f} {row['Precision']:<12.4f} {row['Recall']:<12.4f} {row['F1-Score']:<12.4f} {row['Training_Time']:<10.2f}")

    print("\n" + "="*80)
    print("🏆 BEST MODEL")
    print("="*80)
    best = results_df.iloc[0]
    print(f"Model: {best['Model']}")
    print(f"Accuracy: {best['Accuracy']:.2%} (correct predictions)")
    print(f"Precision: {best['Precision']:.2%} (when it says 'bloom', it's right {best['Precision']:.0%} of the time)")
    print(f"Recall: {best['Recall']:.2%} (catches {best['Recall']:.0%} of actual blooms)")
    print(f"F1-Score: {best['F1-Score']:.4f} (overall balance)")

    print("\n" + "="*80)
    print("💡 INTERPRETATION FOR ROV")
    print("="*80)
    print(f"With 4 sensors, the {best['Model']} can detect blooms with {best['Accuracy']:.0%} accuracy.")
    print(f"This means your ROV will correctly identify bloom conditions {best['Accuracy']*100:.0f} out of 100 times.")
    print("="*80 + "\n")

    # Confusion Matrix for best model
    print("\n" + "="*80)
    print("CONFUSION MATRIX (Best Model)")
    print("="*80)
    # Retrain best model to get predictions
    best_model_name = best['Model']
    if best_model_name == 'Random Forest':
        best_model = rf_model
    elif best_model_name == 'Logistic Regression':
        best_model = lr_model
    elif best_model_name == 'Gradient Boosting':
        best_model = gb_model
    else:
        best_model = nn_model

    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print("\n                Predicted")
    print("              No Bloom  Bloom")
    print(f"Actual No Bloom   {cm[0,0]:<6}  {cm[0,1]:<6}")
    print(f"       Bloom      {cm[1,0]:<6}  {cm[1,1]:<6}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=['No Bloom', 'Bloom']))
    
    

def main():
    #load_and_process_data()
    #train_model()
    #explore_water_chemistry()
    #load_and_process_data_classification(threshold=20)
    train_classification_model()


if __name__ == "__main__":
    main()