import streamlit as st
import lasio
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io
import tempfile
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Lithology Predictor - FORCE 2020 Resilient", layout="wide")

# Paths to models (Assumed to be in 'models/' directory relative to app.py)
MODEL_PATH = "models/xgboost_force_2020_resilient.joblib"
ENCODER_PATH = "models/lithology_label_encoder_resilient.joblib"

# --- MAPPING ---
LITHO_COLORS = {
    'Sandstone': '#F4A460',       # SandyBrown
    'Shale': '#2E8B57',           # SeaGreen
    'Limestone': '#4169E1',       # RoyalBlue
    'Sandstone/Shale': '#BDB76B', # DarkKhaki
    'Marl': '#808000',            # Olive
    'Dolomite': '#800080',        # Purple
    'Chalk': '#87CEEB',           # SkyBlue
    'Coal': '#000000',            # Black
    'Halite': '#FFC0CB',          # Pink
    'Anhydrite': '#FFD700',       # Gold
    'Tuff': '#A9A9A9',            # DarkGray
    'Basement': '#8B4513'         # SaddleBrown
}

LITHO_PT = {
    30000: 'Sandstone',
    65000: 'Shale',
    70000: 'Limestone',
    65030: 'Sandstone/Shale',
    80000: 'Marl',
    74000: 'Dolomite',
    70032: 'Chalk',
    90000: 'Coal',
    88000: 'Halite',
    86000: 'Anhydrite',
    99000: 'Tuff',
    93000: 'Basement',
    
    # Strings just in case
    'Sandstone': 'Sandstone',
    'Shale': 'Shale',
    'Limestone': 'Limestone',
    'Sandstone/Shale': 'Sandstone/Shale',
    'Coal': 'Coal'
}

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return model, le
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

def process_las(las_file_obj):
    # Create a temp file because lasio.read needs a path or string
    with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp:
        tmp.write(las_file_obj.getvalue())
        tmp_path = tmp.name
    
    try:
        las = lasio.read(tmp_path)
        df = las.df().reset_index()
        
        # Handle potential index name variability (DEPT vs DEPTH)
        if 'DEPT' not in df.columns:
            # Check if index is DEPT
            if las.index_unit and 'm' in str(las.index_unit).lower(): # Assuming metric
                 df['DEPT'] = df.index
    finally:
        os.remove(tmp_path)
        
    return df, las

def normalize_curves(df, las_keys):
    # Master Mapping Logic
    curve_map = {
        'DEPT': 'DEPT',
        'GR': None,
        'RHOB': None,
        'NPHI': None,
        'DTC': None,
        'PEF': None,
        'RDEP': None,
        'CALI': None
    }
    
    # Heuristic Mapping
    columns = [c.upper() for c in df.columns]
    df.columns = columns # Upper case all for consistency
    
    # GR
    if 'CGR' in columns: curve_map['GR'] = 'CGR'
    elif 'GR' in columns: curve_map['GR'] = 'GR'
    elif 'GAM' in columns: curve_map['GR'] = 'GAM'
    
    # RHOB
    if 'RHOB' in columns: curve_map['RHOB'] = 'RHOB'
    elif 'RHOZ' in columns: curve_map['RHOB'] = 'RHOZ'
    
    # NPHI
    if 'NPHI' in columns: curve_map['NPHI'] = 'NPHI'
    elif 'TNPH' in columns: curve_map['NPHI'] = 'TNPH'
    elif 'NEU' in columns: curve_map['NPHI'] = 'NEU'
    
    # DTC (Sonic) - The tricky one
    sonic_opts = ['SS1', 'LS', 'SS2', 'DTC', 'DT', 'AC']
    for s in sonic_opts:
        if s in columns:
            curve_map['DTC'] = s
            break
            
    # PEF
    if 'PEF' in columns: curve_map['PEF'] = 'PEF'
    
    # RDEP
    if 'ILD' in columns: curve_map['RDEP'] = 'ILD'
    elif 'LL' in columns: curve_map['RDEP'] = 'LL'
    elif 'RDEP' in columns: curve_map['RDEP'] = 'RDEP'
    
    # CALI
    if 'CALI' in columns: curve_map['CALI'] = 'CALI'
    
    return curve_map

def prepare_for_model(df, curve_map):
    # Create standardized dataframe
    df_model = pd.DataFrame()
    df_model['DEPT'] = df[curve_map['DEPT']]
    
    # Required features by model (even if NaN)
    master_features = ['CALI', 'RDEP', 'DTC', 'NPHI', 'GR', 'RHOB', 'RMED', 'DRHO', 'PEF', 
                      'BS', 'RSHA', 'SP', 'ROP']
    
    mapped_status = {}
    
    for feature in master_features:
        source = curve_map.get(feature)
        if source and source in df.columns:
            series = df[source].copy()
            
            # --- CORRECTIONS ---
            # NPHI
            if feature == 'NPHI' and series.mean() > 1.0:
                series = series / 100.0
                mapped_status['NPHI'] = "Found (Converted % to decimal)"
            elif feature == 'NPHI':
                mapped_status['NPHI'] = "Found"
                
            # DTC
            if feature == 'DTC':
                if source == 'SS1':
                    series = series / 2.0
                    mapped_status['DTC'] = f"Found as {source} (Divided by 2)"
                elif source in ['LS', 'SS2']:
                    series = series / 3.0
                    mapped_status['DTC'] = f"Found as {source} (Divided by 3)"
                else:
                    mapped_status['DTC'] = f"Found as {source}"
                
                # Clip
                series = series.clip(20, 200)
            elif source:
                mapped_status[feature] = f"Found as {source}"
                
            df_model[feature] = series
        else:
            df_model[feature] = np.nan
            mapped_status[feature] = "Not Found (NaN)"
            
    return df_model, mapped_status

def run_prediction(model, le, df_model):
    features = ['CALI', 'RDEP', 'DTC', 'NPHI', 'GR', 'RHOB', 'RMED', 'DRHO', 'PEF', 
                'BS', 'RSHA', 'SP', 'ROP']
    
    # Filter valid rows for prediction (must have at least GR and RHOB to be meaningful)
    # Resilient model can handle some NaNs, but empty rows are useless
    valid_mask = df_model['GR'].notna() & df_model['RHOB'].notna()
    
    if valid_mask.sum() == 0:
        return None, "No valid data (GR + RHOB required) for prediction."
    
    X = df_model.loc[valid_mask, features]
    
    # Predict Probabilities for Sensitivity Analysis
    probs = model.predict_proba(X)
    classes = le.classes_
    
    # Map indices
    # Handle float vs int type mismatch in classes search
    def find_idx(code):
        # classes is typically float32 from parquet
        matches = np.where(classes == code)[0]
        if len(matches) > 0: return matches[0]
        return -1

    idx_sand = find_idx(30000)
    idx_mixed = find_idx(65030)
    idx_shale = find_idx(65000)
    idx_carb = find_idx(70000)
    idx_coal = find_idx(90000)
    
    final_litho = []
    final_litho_name = []
    
    # --- SENSITIVITY CALIBRATION ---
    probs = model.predict_proba(X)
    
    for i in range(len(probs)):
        p_sand = probs[i, idx_sand]
        p_mixed = probs[i, idx_mixed] if idx_mixed != -1 else 0
        p_carb = probs[i, idx_carb] if idx_carb != -1 else 0
        p_coal = probs[i, idx_coal] if idx_coal != -1 else 0
        p_reservoir = p_sand + p_mixed
        
        val_code = None
        val_name = ""

        if p_coal > 0.20: 
            val_name = 'Coal'
            val_code = 90000
        elif p_carb > 0.40: 
            val_name = 'Limestone'
            val_code = 70000
        elif p_reservoir > 0.30: # Adjusted Trigger
            if p_sand > p_mixed:
                val_name = 'Sandstone'
                val_code = 30000
            else:
                val_name = 'Sandstone/Shale'
                val_code = 65030
        else:
            # Fallback to argmax
            val_code = classes[np.argmax(probs[i])]
            # Map code to name using our internal dictionary
            val_name = LITHO_PT.get(int(val_code) if not np.isnan(val_code) else 0, 'Unknown')
            if val_name == 'Unknown':
                 val_name = str(val_code)

        final_litho.append(val_code)
        final_litho_name.append(val_name)
        
    # Create Series with the correct index before assigning
    df_model.loc[valid_mask, 'LITH_CODE'] = pd.Series(final_litho, index=df_model.loc[valid_mask].index)
    df_model.loc[valid_mask, 'LITH_PRED'] = pd.Series(final_litho_name, index=df_model.loc[valid_mask].index)
    
    # Store Probabilities for Visualization
    df_model.loc[valid_mask, 'PROB_SAND'] = pd.Series(probs[:, idx_sand], index=df_model.loc[valid_mask].index)
    df_model.loc[valid_mask, 'PROB_SHALE'] = pd.Series(probs[:, idx_shale], index=df_model.loc[valid_mask].index)
    df_model.loc[valid_mask, 'PROB_LIME'] = pd.Series(probs[:, idx_carb], index=df_model.loc[valid_mask].index)
    df_model.loc[valid_mask, 'PROB_COAL'] = pd.Series(probs[:, idx_coal], index=df_model.loc[valid_mask].index)
    if idx_mixed != -1:
        df_model.loc[valid_mask, 'PROB_MIXED'] = pd.Series(probs[:, idx_mixed], index=df_model.loc[valid_mask].index)
    else:
        df_model.loc[valid_mask, 'PROB_MIXED'] = 0.0
    
    return df_model, None

def plot_results(df):
    # Create a composite log
    df_view = df.dropna(subset=['LITH_PRED']).sort_values('DEPT')
    
    if df_view.empty: return None
    
    min_depth = df_view['DEPT'].min()
    max_depth = df_view['DEPT'].max()
    
    # Increased to 6 columns to add Probability Track
    fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(16, 15), sharey=True)
    
    # GR
    ax[0].plot(df_view['GR'], df_view['DEPT'], 'g', lw=0.5)
    ax[0].set_xlim(0, 150)
    ax[0].set_title("Gamma Ray")
    ax[0].set_ylabel("Depth (m)")
    ax[0].grid(True, alpha=0.5)
    
    # RHOB
    ax[1].plot(df_view['RHOB'], df_view['DEPT'], 'r', lw=0.5)
    ax[1].set_xlim(1.95, 2.95)
    ax[1].set_title("Density")
    ax[1].grid(True, alpha=0.5)
    
    # NPHI
    ax[2].plot(df_view['NPHI'], df_view['DEPT'], 'b', lw=0.5)
    ax[2].set_xlim(0.6, 0)
    ax[2].set_title("Neutron")
    ax[2].grid(True, alpha=0.5)
    
    # DTC
    if df_view['DTC'].notna().sum() > 0:
        ax[3].plot(df_view['DTC'], df_view['DEPT'], 'm', lw=0.5)
        ax[3].set_xlim(200, 40)
    ax[3].set_title("Sonic")
    ax[3].grid(True, alpha=0.5)
    
    # PROBABILITY TRACK (New!)
    # Sand (Orange) + Mixed (Yellow)
    if 'PROB_SAND' in df_view.columns:
        # Area chart for Sand probability
        ax[4].fill_betweenx(df_view['DEPT'], 0, df_view['PROB_SAND'], color='orange', alpha=0.6, label='Sand')
        ax[4].fill_betweenx(df_view['DEPT'], df_view['PROB_SAND'], df_view['PROB_SAND'] + df_view['PROB_MIXED'], color='yellow', alpha=0.4, label='Mixed')
        ax[4].set_xlim(0, 1)
        ax[4].set_title("Sand Prob")
        ax[4].grid(True, alpha=0.5)
    
    # Litho
    for litho, color in LITHO_COLORS.items():
        mask = df_view['LITH_PRED'] == litho
        if mask.any():
            ax[5].scatter(np.ones(mask.sum()), df_view.loc[mask, 'DEPT'], 
                          c=color, s=15, marker='s', label=litho, edgecolors='none')
    
    ax[5].set_xlim(0.5, 1.5)
    ax[5].set_xticks([])
    ax[5].set_title("Lithology")
    # Put legend below
    ax[5].legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize='small')
    
    ax[0].invert_yaxis()
    plt.tight_layout()
    return fig

def plot_lithology_only(df):
    df_view = df.dropna(subset=['LITH_PRED']).sort_values('DEPT')
    if df_view.empty: return None
    
    fig, ax = plt.subplots(figsize=(4, 15))
    
    # Plot as a continuous bar/image would be better, but scatter is robust
    for litho, color in LITHO_COLORS.items():
        mask = df_view['LITH_PRED'] == litho
        if mask.any():
            ax.scatter(np.ones(mask.sum()), df_view.loc[mask, 'DEPT'], 
                          c=color, s=200, marker='_', label=litho, edgecolors='none') # Use marker '_' for bed look
    
    ax.set_ylim(df_view['DEPT'].max(), df_view['DEPT'].min())
    ax.set_xlim(0.9, 1.1)
    ax.set_xticks([])
    ax.set_title("Predicted Lithology")
    ax.set_ylabel("Depth (m)")
    
    # Legend on the side
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

# --- MAIN APP ---

st.title("🏗️ Lithology Predictor AI")
st.markdown("### FORCE 2020 Resilient Model (Calibrated for Potiguar Basin)")

st.info("Upload a standard `.las` file to get automatic lithology classification.")

uploaded_file = st.file_uploader("Choose a LAS file", type=['las'])

if uploaded_file is not None:
    st.write("---")
    st.write("### 1. File Processing")
    
    with st.spinner("Reading LAS file..."):
        df_raw, las_obj = process_las(uploaded_file)
        st.success(f"Loaded **{uploaded_file.name}**. Rows: {len(df_raw)}")
        
    # Model Loading
    model, le = load_model()
    if model is None:
        st.stop()
        
    st.write("### 2. Curve Mapping & Correction")
    curve_map = normalize_curves(df_raw, las_obj.keys())
    
    # Display Mapping
    st.json(curve_map)
    
    df_ready, status = prepare_for_model(df_raw, curve_map)
    
    with st.expander("View Curve Status details"):
        st.write(status)
        
    st.write("### 3. AI Prediction (Inference)")
    
    if st.button("Run Prediction Model"):
        with st.spinner("Running XGBoost Resilient Model..."):
            df_result, err = run_prediction(model, le, df_ready)
            
            if err:
                st.error(err)
            else:
                st.success("Prediction Complete!")
                
                # --- TABS FOR VISUALIZATION ---
                tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📈 Composite Log", "🪨 Lithology Strip"])
                
                with tab1:
                    st.write("#### Lithology Statistics")
                    
                    # Calculate Net Sand (Approx)
                    # Step size calculation
                    step = df_result['DEPT'].diff().median()
                    if np.isnan(step): step = 0.1524 # Standard 0.5ft
                    
                    litho_counts = df_result['LITH_PRED'].value_counts()
                    litho_meters = litho_counts * step
                    
                    # Create nice dataframe
                    stats_df = pd.DataFrame({
                        'Count': litho_counts,
                        'Thickness (m)': litho_meters
                    })
                    stats_df['%'] = (stats_df['Count'] / len(df_result) * 100).map('{:.1f}%'.format)
                    
                    st.dataframe(stats_df)
                    
                    st.write(f"**Estimated Net Sand (Sandstone + Mixed):** {(litho_meters.get('Sandstone', 0) + litho_meters.get('Sandstone/Shale', 0)):.2f} meters")
                    
                    st.bar_chart(litho_counts)

                with tab2:
                    st.write("#### Composite Log (With Probability Track)")
                    fig = plot_results(df_result)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("Not enough data to plot.")
                
                with tab3:
                    st.write("#### Clean Lithology Strip")
                    fig_lith = plot_lithology_only(df_result)
                    if fig_lith:
                        st.pyplot(fig_lith)
                
                # Download
                st.write("### 4. Export Results")
                
                # Convert to CSV
                csv = df_result.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download CSV Result",
                    data=csv,
                    file_name=f"{uploaded_file.name}_lithology.csv",
                    mime='text/csv',
                )