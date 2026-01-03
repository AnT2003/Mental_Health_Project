import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import traceback

# --- CÁC THƯ VIỆN CẦN THIẾT CHO SKLEARN PIPELINE ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder, StandardScaler, OrdinalEncoder

# ==============================================================================
# 0. CẤU HÌNH BẢO MẬT & MÔI TRƯỜNG
# ==============================================================================
# Load biến môi trường từ file .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# Cấu hình Secret Key (Lấy từ .env)
app.secret_key = os.getenv("SECRET_KEY", "default-dev-key-do-not-use-in-prod")

# Lấy các cấu hình khác
PORT = int(os.getenv("PORT", 8080))
ENV_MODE = os.getenv("FLASK_ENV", "production")
DEBUG_MODE = True if ENV_MODE == "development" else False

# ==============================================================================
# 1. ĐỊNH NGHĨA các CLASS featurer transformers 
# ==============================================================================

class LogicalCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_out = X.copy()
        
        # --- 1. CLEAN SLEEP DURATION (Gom nhóm theo ý nghĩa) ---
        if 'Sleep Duration' in X_out.columns:
            def clean_sleep(val):
                s = str(val).lower().strip()
                # Nhóm < 4 hours
                if any(x in s for x in ['1-2', '2-3', '3-4',  '1-3']):
                    return 'Less than 4 hours'
                # Nhóm 4-6 hours
                elif any(x in s for x in ['less than 5','5-6', '4-6', '3-6','4-5','than 5']):
                    return '4-6 hours'
                # Nhóm 7-8 hours (Chuẩn)
                elif any(x in s for x in ['7-8', '6-8', '6-7', '8 hours', '9-5', '10-6']): 
                    return '7-8 hours'
                # Nhóm > 8 hours
                elif any(x in s for x in ['more than 8', '8-9', '9-11', '10-11']):
                    return 'More than 8 hours'
                # Rác -> Unknown
                else:
                    return 'Unknown'
            X_out['Sleep Duration'] = X_out['Sleep Duration'].apply(clean_sleep)

        # --- 2. CLEAN DIETARY HABITS (Gom nhóm theo ý nghĩa) ---
        if 'Dietary Habits' in X_out.columns:
            def clean_diet(val):
                s = str(val).lower().strip()
                if s in ['healthy', 'more healthy']:
                    return 'Healthy'
                elif s in ['moderate']:
                    return 'Moderate'
                elif s in ['unhealthy', 'less than healthy', 'no healthy', 'less healthy']:
                    return 'Unhealthy'
                else: # Rác -> Unknown
                    return 'Unknown'
            X_out['Dietary Habits'] = X_out['Dietary Habits'].apply(clean_diet)

        # --- 3. GỘP CỘT CHO MODEL ---
        if 'Profession' in X_out.columns and 'Degree' in X_out.columns:
            X_out['Occupation'] = X_out['Profession'].fillna(X_out['Degree'])
            
        if 'Work Pressure' in X_out.columns and 'Academic Pressure' in X_out.columns:
            X_out['Pressure'] = X_out['Work Pressure'].fillna(X_out['Academic Pressure'])

        if 'Job Satisfaction' in X_out.columns and 'Study Satisfaction' in X_out.columns:
            X_out['Satisfaction'] = X_out['Job Satisfaction'].fillna(X_out['Study Satisfaction'])
            
        return X_out

class RareLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, threshold=5):
        self.variables = variables or []
        self.threshold = threshold
        self.valid_labels_ = {} 

    def fit(self, X, y=None):
        for col in self.variables:
            if col in X.columns:
                counts = X[col].value_counts()
                self.valid_labels_[col] = counts[counts > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.variables:
            if col in X_out.columns:
                valid_list = self.valid_labels_.get(col, [])
                # Gom giá trị hiếm thành 'Other'
                X_out[col] = np.where(X_out[col].isin(valid_list), X_out[col], 'Other')
        return X_out

# ==============================================================================
# 2. KHỞI TẠO
# ==============================================================================

DATA_PATH = 'train.csv'
MODEL_PATH = 'depression_prediction_system.pkl'
CONFIG_PATH = 'model_ui_config.json'

system_bundle = None
ui_config = None

# --- LOAD MODEL ---
if os.path.exists(MODEL_PATH):
    try:
        system_bundle = joblib.load(MODEL_PATH)
        print(f">>> MODEL LOADED SUCCESSFULLY: {MODEL_PATH}")
        print(f">>> Threshold Setting: {system_bundle.get('threshold', 0.5)}")
    except Exception as e:
        print(f"!!! CRITICAL ERROR: Không thể load model. Chi tiết: {e}")

# --- LOAD UI CONFIG ---
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            ui_config = json.load(f)
        print(">>> UI CONFIG LOADED")
    except Exception as e:
        print(f"!!! Error loading config: {e}")

def load_data():
    """Hàm load và làm sạch sơ bộ dữ liệu cho Dashboard"""
    if not os.path.exists(DATA_PATH): return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)

    # 1. Áp dụng LogicalCleaner (Xử lý chuỗi Sleep/Diet, tạo cột gộp)
    cleaner = LogicalCleaner()
    df = cleaner.transform(df)
    # 2. Áp dụng Rare Label Removal (Thủ công cho Dashboard để chart gọn)
    vars_to_clean = ['Degree', 'Profession', 'City', 'Dietary Habits', 'Sleep Duration']
    for col in vars_to_clean:
        if col in df.columns:
            counts = df[col].value_counts()
            valid_values = counts[counts > 5].index
            df[col] = np.where(df[col].isin(valid_values), df[col], 'Other')
    # 3. Convert numeric columns
    cols = ['Work/Study Hours', 'CGPA', 'Work Pressure', 'Academic Pressure', 
            'Job Satisfaction', 'Study Satisfaction', 'Financial Stress', 'Age', 'Depression']
    for c in cols:
        if c in df.columns: 
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
    return df

# ==============================================================================
# 3. ROUTES Trang chủ & Giao diện
# ==============================================================================
@app.route('/')
def home():
    df = load_data()
    if df.empty: return render_template('index.html', data_sample=[], stats={}, column_names=[])
    
    data_sample = df.head(500).to_dict(orient='records')
    column_names = df.columns.tolist()
    
    desc_df = df.describe().reset_index()
    desc_data = desc_df.to_dict(orient='records')
    desc_columns = desc_df.columns.tolist()

    stats = {
        'avg_age': round(df['Age'].mean(), 1),
        'avg_cgpa': round(df['CGPA'].mean(), 2),
        'depression_rate': round((df['Depression'].sum() / len(df)) * 100, 1)
    }
    return render_template('index.html', data_sample=data_sample, stats=stats, column_names=column_names, desc_data=desc_data, desc_columns=desc_columns)

@app.route('/dashboard')
def dashboard(): 
    return render_template('dashboard.html')

@app.route('/ml')
def ml_page():
    return render_template('ml.html')

# ==============================================================================
# 4. API MACHINE LEARNING (PREDICT)
# ==============================================================================

@app.route('/api/model-config', methods=['GET'])
def get_model_config():
    if ui_config:
        return jsonify(ui_config)
    return jsonify({"error": "Config file not found"}), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    if not system_bundle:
        return jsonify({'error': 'Model chưa sẵn sàng.'}), 500

    try:
        data = request.json
        input_df = pd.DataFrame([data])

        # --- 1. Xử lý gộp student và professional ---
        status = data.get('Working Professional or Student')
        if status == 'Student':
            input_df['Work Pressure'] = np.nan
            input_df['Job Satisfaction'] = np.nan
            input_df['Profession'] = np.nan
        elif status == 'Working Professional':
            input_df['Academic Pressure'] = np.nan
            input_df['Study Satisfaction'] = np.nan
            input_df['Degree'] = np.nan
            input_df['CGPA'] = np.nan
        
        # --- 2. Convert kiểu dữ liệu ---
        numeric_cols = ['Age', 'Work/Study Hours', 'Financial Stress', 'CGPA', 'Work Pressure', 'Academic Pressure', 'Job Satisfaction', 'Study Satisfaction']
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # --- 3. CHẠY PIPELINE ---
        cleaner = system_bundle['selector_pipeline']
        X_clean = cleaner.transform(input_df)
        
        required_features = system_bundle['required_features']
        for col in required_features:
            if col not in X_clean.columns: X_clean[col] = 0
        X_selected = X_clean[required_features]
        
        processor = system_bundle['preprocessor']
        X_processed = processor.transform(X_selected)
        
        model = system_bundle['model']
        prob = float(model.predict_proba(X_processed)[:, 1][0])
        threshold = float(system_bundle.get('threshold', 0.5))
        
        if prob >= threshold:
            prediction_text = "Critical (High Risk)"
            recommendation = "Your results exceed the critical threshold. Indicates a high likelihood of depression. Immediate professional support is recommended."
        elif prob >= 0.6:
            prediction_text = "Warning (Elevated Risk)"
            recommendation = "You are below the critical threshold but showing significant signs of stress (> 60%). Pay attention to your mental well-being."
        elif prob >= 0.4:
            prediction_text = "Moderate (Mild Signs)"
            recommendation = "You are in a safe range but showing mild signs of stress (40-60%). Consider some relaxation techniques."
        else:
            prediction_text = "Normal (Low Risk)"
            recommendation = "Your mental health indicators are well within the safe zone (< 40%). Maintain your positive lifestyle."

        prediction = 1 if prob >= threshold else 0

        return jsonify({
            'probability': round(prob * 100, 2),
            'prediction': int(prediction),
            'threshold': round(threshold, 4),
            'risk_level': prediction_text,
            'recommendation': recommendation
        })

    except Exception as e:
        traceback.print_exc() 
        return jsonify({'error': str(e)}), 500

# ==============================================================================
# 5. API DASHBOARD
# ==============================================================================
@app.route('/api/custom-dashboard')
def get_custom_dashboard():
    try:
        df = load_data()
        if df.empty: return jsonify({})

        f_status = request.args.get('status')
        f_gender = request.args.get('gender')
        f_age_min = request.args.get('ageMin')
        f_age_max = request.args.get('ageMax')
        f_history = request.args.get('history')
        f_suicide = request.args.get('suicide')
        f_depression = request.args.get('depression')
        f_degree = request.args.get('degree')
        f_profession = request.args.get('profession')

        if f_status: df = df[df['Working Professional or Student'] == f_status]
        if f_gender: df = df[df['Gender'] == f_gender]
        if f_age_min and f_age_max: df = df[(df['Age'] >= int(f_age_min)) & (df['Age'] <= int(f_age_max))]
        if f_history: df = df[df['Family History of Mental Illness'] == f_history]
        if f_suicide: df = df[df['Have you ever had suicidal thoughts ?'] == f_suicide]
        if f_depression: df = df[df['Depression'] == int(f_depression)]
        if f_degree:
            degree_list = [x.strip() for x in f_degree.split(',') if x.strip()]
            if degree_list: df = df[df['Degree'].isin(degree_list)]
        if f_profession:
            prof_list = [x.strip() for x in f_profession.split(',') if x.strip()]
            if prof_list: df = df[df['Profession'].isin(prof_list)]

        df_global = df

        def safe_int_list(s): return [int(x) for x in s.fillna(0).tolist()]
        def safe_float(v): 
            try: return float(v)
            except: return 0.0
        def safe_str_list(idx): return [str(x) for x in idx.tolist()]

        total_people = len(df_global)
        if total_people == 0:
            return jsonify({'kpi': {'depression_rate': 0}})

        depression_rate = round((int(df_global['Depression'].sum()) / total_people) * 100, 2)
        
        pressure_cols = ['Work Pressure','Academic Pressure']
        satisfaction_cols = ['Job Satisfaction','Study Satisfaction']
        actual_press = [c for c in pressure_cols if c in df_global.columns]
        actual_sat = [c for c in satisfaction_cols if c in df_global.columns]
        
        avg_total_pressure = 0
        if actual_press: avg_total_pressure = round(df_global[actual_press].sum(axis=1).mean(), 2)
        avg_total_satisfaction = 0
        if actual_sat: avg_total_satisfaction = round(df_global[actual_sat].sum(axis=1).mean(), 2)

        total_family_history = int(df_global[df_global['Family History of Mental Illness'] == 'Yes'].shape[0])
        total_suicidal_thoughts = int(df_global[df_global['Have you ever had suicidal thoughts ?'] == 'Yes'].shape[0])

        gender_stats = df_global['Gender'].value_counts()
        status_stats = df_global['Working Professional or Student'].value_counts()
        
        age_groups = df_global.groupby(['Age', 'Depression'], observed=False).size().unstack(fill_value=0)
        if not df_global['Age'].empty:
            min_a, max_a = int(df_global['Age'].min()), int(df_global['Age'].max())
            full_range = range(min_a, max_a + 1)
            age_groups = age_groups.reindex(full_range, fill_value=0)
        
        age_labels_sparse = []
        for age in age_groups.index:
            if age >= 18 and (age - 18) % 3 == 0:
                age_labels_sparse.append(str(age))
            else:
                age_labels_sparse.append("")
        for c in [0, 1]: 
            if c not in age_groups.columns: age_groups[c] = 0

        df_student = df_global[df_global['Working Professional or Student'] == 'Student']
        scatter_data = []
        if not df_student.empty:
            sample_student = df_student.sample(n=min(300, len(df_student)))
            scatter_data = [{'x': safe_float(r['Work/Study Hours']), 'y': safe_float(r['CGPA']), 'r': safe_float(r['Academic Pressure'])*2} for _, r in sample_student.iterrows()]

        levels = [0, 1, 2, 3, 4, 5]
        work_rates = [round(df_global[df_global['Work Pressure'] == l]['Depression'].mean() * 100, 1) if not df_global[df_global['Work Pressure'] == l].empty else 0 for l in levels]
        acad_rates = [round(df_global[df_global['Academic Pressure'] == l]['Depression'].mean() * 100, 1) if not df_global[df_global['Academic Pressure'] == l].empty else 0 for l in levels]

        def get_ranked_tree(group_col, press_col, sat_col):
            if group_col not in df_global.columns: return []
            counts = df_global[group_col].value_counts()
            valid = counts[counts >= 5].index
            df_v = df_global[df_global[group_col].isin(valid)]
            if df_v.empty: return []
            if press_col not in df_v.columns or sat_col not in df_v.columns: return []
            top = df_v.groupby(group_col)[press_col].mean().sort_values(ascending=False).head(5)
            res = []
            for name, p in top.items():
                s = df_v[df_v[group_col] == name][sat_col].mean()
                res.append({'category': str(name), 'type': 'Áp Lực', 'value': round(p, 2)})
                res.append({'category': str(name), 'type': 'Hài Lòng', 'value': round(s, 2)})
            return res

        sleep_stats = pd.DataFrame()
        if 'Sleep Duration' in df_global.columns:
            sleep_stats = df_global.groupby(['Sleep Duration', 'Depression']).size().unstack(fill_value=0)
            for c in [0, 1]: 
                if c not in sleep_stats.columns: sleep_stats[c] = 0
            sleep_stats['Total'] = sleep_stats.sum(axis=1)
            sleep_stats = sleep_stats.sort_values('Total', ascending=False)

        diet_pct = pd.DataFrame()
        if 'Dietary Habits' in df_global.columns:
            diet_counts = df_global.groupby(['Dietary Habits', 'Depression']).size().unstack(fill_value=0)
            for c in [0, 1]: 
                if c not in diet_counts.columns: diet_counts[c] = 0
            diet_pct = diet_counts.div(diet_counts.sum(axis=1), axis=0) * 100

        fin_gr = df_global.groupby('Financial Stress')
        stress_lvls = [1, 2, 3, 4, 5]
        fin_tot = fin_gr.size().reindex(stress_lvls, fill_value=0)
        fin_dep = fin_gr['Depression'].sum().reindex(stress_lvls, fill_value=0)
        fin_sui = df_global[df_global['Have you ever had suicidal thoughts ?'] == 'Yes'].groupby('Financial Stress').size().reindex(stress_lvls, fill_value=0)

        return jsonify({
            'kpi': {'depression_rate': depression_rate, 'avg_total_pressure': avg_total_pressure, 'avg_total_satisfaction': avg_total_satisfaction, 'total_family_history': total_family_history, 'total_suicidal_thoughts': total_suicidal_thoughts},
            'pie_gender': {'labels': safe_str_list(gender_stats.index), 'data': safe_int_list(gender_stats)},
            'pie_status': {'labels': safe_str_list(status_stats.index), 'data': safe_int_list(status_stats)},
            'hist_age': {'labels': age_labels_sparse, 'normal': safe_int_list(age_groups[0]), 'depressed': safe_int_list(age_groups[1])},
            'scatter': scatter_data,
            'comparison_pressure': {'labels': [f'Mức {i}' for i in levels], 'work': work_rates, 'academic': acad_rates},
            'tree_prof': get_ranked_tree('Profession', 'Work Pressure', 'Job Satisfaction'),
            'tree_degree': get_ranked_tree('Degree', 'Academic Pressure', 'Study Satisfaction'),
            'col_suicide': {'labels': safe_str_list(status_stats.index), 'normal': safe_int_list(df_global[df_global['Depression']==0]['Working Professional or Student'].value_counts().reindex(status_stats.index)), 'depressed': safe_int_list(df_global[df_global['Depression']==1]['Working Professional or Student'].value_counts().reindex(status_stats.index))},
            'bar_diet': {'labels': safe_str_list(diet_pct.index) if not diet_pct.empty else [], 'normal': diet_pct[0].round(1).tolist() if not diet_pct.empty else [], 'depressed': diet_pct[1].round(1).tolist() if not diet_pct.empty else []},
            'area_sleep': {'labels': safe_str_list(sleep_stats.index) if not sleep_stats.empty else [], 'normal': safe_int_list(sleep_stats[0]) if not sleep_stats.empty else [], 'depressed': safe_int_list(sleep_stats[1]) if not sleep_stats.empty else []},
            'area_finance': {'labels': [f"Mức {i}" for i in stress_lvls], 'total': safe_int_list(fin_tot), 'depressed': safe_int_list(fin_dep), 'suicide': safe_int_list(fin_sui)}
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Chạy server với biến môi trường PORT
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG_MODE)