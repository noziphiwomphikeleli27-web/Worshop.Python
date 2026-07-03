"""
Universal AI Data Explorer & Machine Learning Workbench
Production-quality Streamlit application for automated data analysis and ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_auc_score,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import joblib
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Universal AI Data Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #2ca02c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_column_types(df):
    """Detect column types automatically"""
    types = {
        'numeric': list(df.select_dtypes(include=[np.number]).columns),
        'categorical': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime': list(df.select_dtypes(include=['datetime64']).columns),
        'boolean': list(df.select_dtypes(include=['bool']).columns),
        'identifier': []
    }
    
    # Detect potential identifier columns
    for col in types['numeric']:
        if df[col].nunique() == len(df) and df[col].dtype in ['int64', 'int32']:
            types['identifier'].append(col)
            if col in types['numeric']:
                types['numeric'].remove(col)
    
    return types

def detect_outliers_iqr(df, column, threshold=1.5):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    return mask.sum(), lower_bound, upper_bound

def suggest_best_model(X_train, y_train, problem_type):
    """Suggest best model based on data characteristics"""
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    suggestions = []
    
    if problem_type == 'classification':
        if n_samples < 100:
            suggestions.append(('KNN', KNeighborsClassifier(), 'Best for small datasets'))
        if n_features < 20:
            suggestions.append(('Logistic Regression', LogisticRegression(max_iter=1000), 'Simple and interpretable'))
        if n_samples > 100 and n_features > 10:
            suggestions.append(('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42), 'Handles complex relationships'))
        if n_samples > 50:
            suggestions.append(('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42), 'High accuracy'))
        if n_features < 50:
            suggestions.append(('Decision Tree', DecisionTreeClassifier(random_state=42), 'Interpretable'))
        if n_samples > 100 and n_features < 100:
            suggestions.append(('SVM', SVC(probability=True, random_state=42), 'Good for high-dimensional'))
    else:  # regression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        suggestions.append(('Linear Regression', LinearRegression(), 'Simple and interpretable'))
        suggestions.append(('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42), 'Handles non-linear relationships'))
        suggestions.append(('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42), 'High accuracy'))
    
    return suggestions

# ============================================================================
# EDA FUNCTIONS
# ============================================================================

def perform_eda(df):
    """Perform comprehensive EDA"""
    results = {}
    
    # Basic info
    results['shape'] = df.shape
    results['columns'] = list(df.columns)
    results['dtypes'] = df.dtypes.to_dict()
    results['missing'] = df.isnull().sum().to_dict()
    results['missing_percent'] = (df.isnull().sum() / len(df) * 100).to_dict()
    results['duplicates'] = df.duplicated().sum()
    
    # Column types
    results['column_types'] = detect_column_types(df)
    
    # Numerical statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        results['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Correlations
        results['correlation'] = df[numeric_cols].corr().to_dict()
        
        # Outliers
        results['outliers'] = {}
        for col in numeric_cols:
            count, lower, upper = detect_outliers_iqr(df, col)
            results['outliers'][col] = {
                'count': count,
                'percentage': count / len(df) * 100,
                'lower_bound': lower,
                'upper_bound': upper
            }
    
    # Categorical statistics
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        results['categorical_stats'] = {}
        for col in cat_cols:
            results['categorical_stats'][col] = {
                'unique': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
    
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_correlation_heatmap(df, columns=None):
    """Create correlation heatmap"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    if len(columns) < 2:
        return go.Figure()
    
    corr = df[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=corr.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        width=800,
        height=600,
        xaxis=dict(tickangle=45)
    )
    
    return fig

def create_distribution_plot(df, column):
    """Create distribution plot with histogram and KDE"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[column].dropna(),
        nbinsx=30,
        name='Histogram',
        marker_color='blue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f'Distribution of {column}',
        xaxis_title=column,
        yaxis_title='Frequency',
        width=700,
        height=500,
        showlegend=True
    )
    
    return fig

def create_boxplot(df, x, y):
    """Create boxplot"""
    fig = go.Figure()
    
    categories = df[x].unique()
    for cat in categories:
        fig.add_trace(go.Box(
            y=df[df[x] == cat][y].dropna(),
            name=str(cat),
            boxmean='sd'
        ))
    
    fig.update_layout(
        title=f'Boxplot of {y} by {x}',
        xaxis_title=x,
        yaxis_title=y,
        width=700,
        height=500
    )
    
    return fig

def create_scatter_plot(df, x, y, color=None):
    """Create scatter plot"""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        title=f'{y} vs {x}',
        trendline='ols',
        hover_data=df.columns.tolist()
    )
    
    fig.update_layout(width=700, height=500)
    return fig

def create_time_series(df, x, y, color=None):
    """Create time series plot"""
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        title=f'{y} over {x}',
        markers=True
    )
    
    fig.update_layout(width=800, height=500)
    return fig

def create_confusion_matrix(y_true, y_pred, labels):
    """Create confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        width=600,
        height=500,
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    
    return fig

def create_feature_importance_plot(features, importance):
    """Create feature importance plot"""
    sorted_idx = np.argsort(importance)
    features_sorted = [features[i] for i in sorted_idx]
    importance_sorted = [importance[i] for i in sorted_idx]
    
    fig = go.Figure(data=go.Bar(
        x=importance_sorted,
        y=features_sorted,
        orientation='h',
        marker_color='skyblue',
        text=importance_sorted,
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        width=700,
        height=500
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<div class="main-header">🧠 Universal AI Data Explorer & ML Workbench</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.title("Navigation")
        
        tabs = st.radio(
            "Select Section",
            ["📊 Data Upload", "🔍 EDA", "⚙️ Preprocessing", "🤖 Model Training", "📈 Results", "💾 Download"]
        )
    
    # ========================================================================
    # DATA UPLOAD
    # ========================================================================
    if tabs == "📊 Data Upload":
        st.header("📊 Data Upload")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your dataset",
                type=['csv', 'xlsx', 'xls', 'tsv'],
                help="Supported formats: CSV, Excel, TSV"
            )
            
            if uploaded_file is not None:
                try:
                    # Detect file type
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_type == 'csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_type in ['xlsx', 'xls']:
                        df = pd.read_excel(uploaded_file)
                    elif file_type in ['tsv', 'txt']:
                        df = pd.read_csv(uploaded_file, delimiter='\t')
                    else:
                        st.error("Unsupported file format")
                        return
                    
                    st.session_state.df = df
                    st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
                    
                    # Display preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(10))
                    
                    # Basic info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    with col4:
                        st.metric("Duplicates", df.duplicated().sum())
                    
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        with col2:
            st.subheader("📁 Sample Data")
            if st.button("Load Sample Dataset"):
                # Create sample data similar to your production data
                sample_data = {
                    'Year': list(range(2015, 2025)) * 5,
                    'Factory': ['A']*10 + ['B']*10 + ['C']*10 + ['D']*10 + ['E']*10,
                    'Labor (Workers)': [80+i*2 for i in range(10)] + [90+i*2 for i in range(10)] + 
                                      [100+i*2 for i in range(10)] + [110+i*2 for i in range(10)] + 
                                      [120+i*2 for i in range(10)],
                    'Capital ($)': [500000+i*25000 for i in range(10)] + [540000+i*25000 for i in range(10)] +
                                  [580000+i*25000 for i in range(10)] + [620000+i*25000 for i in range(10)] +
                                  [660000+i*25000 for i in range(10)],
                    'Raw Materials (tons)': [900+i*20 for i in range(10)] + [960+i*20 for i in range(10)] +
                                           [1020+i*20 for i in range(10)] + [1080+i*20 for i in range(10)] +
                                           [1140+i*20 for i in range(10)],
                    'Technology Index': [70+i*2 for i in range(10)] + [73+i*2 for i in range(10)] +
                                       [76+i*2 for i in range(10)] + [79+i*2 for i in range(10)] +
                                       [82+i*2 for i in range(10)],
                    'Energy (MWh)': [1200+i*25 for i in range(10)] + [1280+i*25 for i in range(10)] +
                                   [1360+i*25 for i in range(10)] + [1440+i*25 for i in range(10)] +
                                   [1520+i*25 for i in range(10)],
                    'Production Output (Units)': [12340+i*296 for i in range(10)] + [13180+i*296 for i in range(10)] +
                                                [14020+i*296 for i in range(10)] + [14860+i*296 for i in range(10)] +
                                                [15700+i*296 for i in range(10)]
                }
                df = pd.DataFrame(sample_data)
                
                # Add Production Rate
                df['Production Rate'] = df['Production Output (Units)'].apply(
                    lambda x: 'High' if x > 15000 else ('Medium' if x >= 12000 else 'Low')
                )
                
                st.session_state.df = df
                st.success("✅ Sample dataset loaded!")
                st.rerun()
    
    # ========================================================================
    # EDA
    # ========================================================================
    elif tabs == "🔍 EDA":
        st.header("🔍 Exploratory Data Analysis")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Run EDA
            with st.spinner("Analyzing data..."):
                eda_results = perform_eda(df)
            
            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Rows", eda_results['shape'][0])
            with col2:
                st.metric("Columns", eda_results['shape'][1])
            with col3:
                st.metric("Missing %", f"{sum(eda_results['missing_percent'].values())/len(df.columns):.1f}%")
            with col4:
                st.metric("Duplicates", eda_results['duplicates'])
            with col5:
                st.metric("Data Quality", f"{100 - sum(eda_results['missing_percent'].values())/len(df.columns):.0f}%")
            
            # Column types
            st.subheader("📋 Column Types")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Numeric", len(eda_results['column_types']['numeric']))
            with col2:
                st.metric("Categorical", len(eda_results['column_types']['categorical']))
            with col3:
                st.metric("Identifier", len(eda_results['column_types']['identifier']))
            
            # Missing values
            if sum(eda_results['missing'].values()) > 0:
                st.subheader("⚠️ Missing Values")
                missing_df = pd.DataFrame({
                    'Column': list(eda_results['missing'].keys()),
                    'Missing': list(eda_results['missing'].values()),
                    'Percentage': list(eda_results['missing_percent'].values())
                })
                missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
                st.dataframe(missing_df)
            
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.subheader("📊 Correlation Analysis")
                fig = create_correlation_heatmap(df)
                st.plotly_chart(fig, use_container_width=True)
            
            # Outliers
            if 'outliers' in eda_results:
                st.subheader("🔍 Outlier Detection")
                outlier_df = pd.DataFrame({
                    'Column': list(eda_results['outliers'].keys()),
                    'Outliers': [v['count'] for v in eda_results['outliers'].values()],
                    'Percentage': [v['percentage'] for v in eda_results['outliers'].values()]
                })
                outlier_df = outlier_df[outlier_df['Outliers'] > 0].sort_values('Outliers', ascending=False)
                if len(outlier_df) > 0:
                    st.dataframe(outlier_df)
                else:
                    st.info("No outliers detected using IQR method")
            
            # Distributions
            st.subheader("📈 Distributions")
            selected_col = st.selectbox("Select column to visualize", numeric_cols)
            if selected_col:
                fig = create_distribution_plot(df, selected_col)
                st.plotly_chart(fig, use_container_width=True)
            
            # Time series if Year column exists
            if 'Year' in df.columns and len(numeric_cols) > 0:
                st.subheader("📈 Time Series Analysis")
                y_col = st.selectbox("Select variable to plot over time", numeric_cols)
                if y_col:
                    color_col = None
                    if 'Factory' in df.columns:
                        color_col = 'Factory'
                    fig = create_time_series(df, 'Year', y_col, color_col)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Target suggestion
            st.subheader("🎯 Target Variable Suggestion")
            suggestions = []
            for col in numeric_cols:
                if col not in eda_results['column_types']['identifier']:
                    nunique = df[col].nunique()
                    if 2 <= nunique <= 20:
                        suggestions.append(f"{col} (Classification)")
                    elif nunique > 20:
                        suggestions.append(f"{col} (Regression)")
            
            if suggestions:
                st.info("Potential target variables:")
                for s in suggestions:
                    st.write(f"- {s}")
        else:
            st.warning("Please upload a dataset first")
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    elif tabs == "⚙️ Preprocessing":
        st.header("⚙️ Data Preprocessing")
        
        if st.session_state.df is not None:
            df = st.session_state.df.copy()
            
            st.subheader("🔧 Preprocessing Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Handle missing values
                st.markdown("### Missing Values")
                missing_strategy = st.selectbox(
                    "Imputation Strategy",
                    ['None', 'Mean', 'Median', 'Mode', 'Drop', 'Constant']
                )
                
                if missing_strategy == 'Constant':
                    constant_value = st.text_input("Constant Value", "0")
            
            with col2:
                # Scaling
                st.markdown("### Feature Scaling")
                scaling_method = st.selectbox(
                    "Scaling Method",
                    ['None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler']
                )
            
            # Encoding
            st.markdown("### Encoding")
            encoding_method = st.selectbox(
                "Categorical Encoding",
                ['None', 'One-Hot', 'Label Encoding']
            )
            
            # Target selection
            st.markdown("### Target Selection")
            all_cols = df.columns.tolist()
            st.session_state.target_col = st.selectbox("Select Target Column", all_cols)
            
            # Feature selection
            if st.session_state.target_col:
                feature_cols = [col for col in all_cols if col != st.session_state.target_col]
                st.session_state.feature_cols = st.multiselect(
                    "Select Feature Columns",
                    feature_cols,
                    default=feature_cols
                )
            
            # Apply preprocessing
            if st.button("🔄 Apply Preprocessing"):
                with st.spinner("Preprocessing data..."):
                    processed_df = df.copy()
                    
                    # Handle missing values
                    if missing_strategy != 'None':
                        if missing_strategy == 'Drop':
                            processed_df = processed_df.dropna()
                        else:
                            strategy = missing_strategy.lower()
                            if missing_strategy == 'Constant':
                                imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
                            else:
                                imputer = SimpleImputer(strategy=strategy)
                            processed_df[processed_df.columns] = imputer.fit_transform(processed_df[processed_df.columns])
                    
                    # Encode categorical
                    if encoding_method != 'None':
                        cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns
                        if encoding_method == 'One-Hot':
                            processed_df = pd.get_dummies(processed_df, columns=cat_cols)
                        elif encoding_method == 'Label Encoding':
                            for col in cat_cols:
                                processed_df[col] = LabelEncoder().fit_transform(processed_df[col].astype(str))
                    
                    # Scale features
                    if scaling_method != 'None' and st.session_state.feature_cols:
                        numeric_cols = [col for col in st.session_state.feature_cols 
                                      if col in processed_df.columns and 
                                      processed_df[col].dtype in ['float64', 'int64']]
                        
                        if numeric_cols:
                            if scaling_method == 'StandardScaler':
                                scaler = StandardScaler()
                            elif scaling_method == 'MinMaxScaler':
                                scaler = MinMaxScaler()
                            elif scaling_method == 'RobustScaler':
                                scaler = RobustScaler()
                            
                            processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
                            st.session_state.scaler = scaler
                    
                    st.session_state.processed_df = processed_df
                    st.success(f"✅ Preprocessing complete! Shape: {processed_df.shape}")
                    
                    # Show preview
                    st.subheader("📋 Preprocessed Data Preview")
                    st.dataframe(processed_df.head(10))
        else:
            st.warning("Please upload a dataset first")
    
    # ========================================================================
    # MODEL TRAINING
    # ========================================================================
    elif tabs == "🤖 Model Training":
        st.header("🤖 Model Training & Evaluation")
        
        if st.session_state.processed_df is not None:
            df = st.session_state.processed_df
            
            if st.session_state.target_col and st.session_state.feature_cols:
                # Prepare data
                X = df[st.session_state.feature_cols]
                y = df[st.session_state.target_col]
                
                # Encode target if categorical
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    st.session_state.encoder = le
                    problem_type = 'classification'
                    classes = le.classes_
                else:
                    y_encoded = y
                    problem_type = 'regression' if y.nunique() > 20 else 'classification'
                    classes = np.unique(y_encoded)
                
                # Split data
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
                random_state = st.number_input("Random State", 0, 100, 42)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state,
                    stratify=y_encoded if problem_type == 'classification' and len(np.unique(y_encoded)) > 1 else None
                )
                
                st.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
                
                # Model selection
                st.subheader("🎯 Model Selection")
                
                if problem_type == 'classification':
                    models = {
                        'KNN': KNeighborsClassifier(),
                        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
                        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
                        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
                        'SVM': SVC(probability=True, random_state=random_state)
                    }
                else:
                    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                    from sklearn.linear_model import LinearRegression, Ridge, Lasso
                    models = {
                        'Linear Regression': LinearRegression(),
                        'Ridge': Ridge(random_state=random_state),
                        'Lasso': Lasso(random_state=random_state),
                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
                    }
                
                selected_model = st.selectbox("Select Model", list(models.keys()))
                model = models[selected_model]
                
                # Hyperparameter tuning for KNN
                if selected_model == 'KNN':
                    st.subheader("🔧 KNN Hyperparameter Tuning")
                    tune_knn = st.checkbox("Tune KNN Parameters")
                    
                    if tune_knn:
                        k_range = st.slider("K Range", 1, 30, (1, 20))
                        weights = st.multiselect("Weights", ['uniform', 'distance'], ['uniform', 'distance'])
                        metrics = st.multiselect("Metrics", ['euclidean', 'manhattan', 'minkowski'], ['euclidean'])
                        
                        param_grid = {
                            'n_neighbors': list(range(k_range[0], k_range[1]+1)),
                            'weights': weights,
                            'metric': metrics
                        }
                        
                        if st.button("🔍 Find Best Parameters"):
                            with st.spinner("Performing Grid Search..."):
                                grid_search = GridSearchCV(
                                    KNeighborsClassifier(),
                                    param_grid,
                                    cv=5,
                                    scoring='accuracy',
                                    n_jobs=-1
                                )
                                grid_search.fit(X_train, y_train)
                                
                                st.success(f"Best Parameters: {grid_search.best_params_}")
                                st.metric("Best CV Accuracy", f"{grid_search.best_score_:.4f}")
                                
                                model = grid_search.best_estimator_
                                st.session_state.model = model
                
                # Train model
                if st.button("🚀 Train Model"):
                    with st.spinner("Training model..."):
                        # Handle KNN parameter tuning if not done
                        if selected_model == 'KNN' and not tune_knn:
                            # Scale for KNN
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            st.session_state.scaler = scaler
                            
                            # Train KNN
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                            y_train_pred = model.predict(X_train_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            y_train_pred = model.predict(X_train)
                        
                        st.session_state.model = model
                        st.session_state.results = {
                            'model': model,
                            'model_name': selected_model,
                            'y_pred': y_pred,
                            'y_train_pred': y_train_pred,
                            'y_test': y_test,
                            'y_train': y_train,
                            'X_test': X_test,
                            'X_train': X_train,
                            'problem_type': problem_type,
                            'classes': classes if problem_type == 'classification' else None
                        }
                        
                        st.success("✅ Model trained successfully!")
                        
                        # Display results
                        display_model_results(st.session_state.results)
            else:
                st.warning("Please select target and feature columns in the Preprocessing tab")
        else:
            st.warning("Please preprocess the data first")
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    elif tabs == "📈 Results":
        st.header("📈 Model Results")
        
        if st.session_state.results:
            display_model_results(st.session_state.results)
        else:
            st.warning("No model results found. Please train a model first.")
    
    # ========================================================================
    # DOWNLOAD
    # ========================================================================
    elif tabs == "💾 Download":
        st.header("💾 Download Results")
        
        if st.session_state.results:
            st.subheader("📥 Download Options")
            
            # Download predictions
            if 'y_pred' in st.session_state.results:
                pred_df = pd.DataFrame({
                    'Actual': st.session_state.results['y_test'],
                    'Predicted': st.session_state.results['y_pred']
                })
                
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    "📊 Download Predictions (CSV)",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )
            
            # Download model
            if st.session_state.model:
                model_bytes = io.BytesIO()
                joblib.dump(st.session_state.model, model_bytes)
                st.download_button(
                    "🤖 Download Model (Joblib)",
                    model_bytes.getvalue(),
                    "model.pkl",
                    "application/octet-stream"
                )
            
            # Download scaler
            if st.session_state.scaler:
                scaler_bytes = io.BytesIO()
                joblib.dump(st.session_state.scaler, scaler_bytes)
                st.download_button(
                    "📐 Download Scaler",
                    scaler_bytes.getvalue(),
                    "scaler.pkl",
                    "application/octet-stream"
                )
            
            # Download preprocessed data
            if st.session_state.processed_df is not None:
                csv = st.session_state.processed_df.to_csv(index=False)
                st.download_button(
                    "📊 Download Preprocessed Data",
                    csv,
                    "preprocessed_data.csv",
                    "text/csv"
                )
        else:
            st.warning("No results to download. Train a model first.")

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_model_results(results):
    """Display model results with metrics and visualizations"""
    
    model_name = results.get('model_name', 'Model')
    problem_type = results.get('problem_type', 'classification')
    y_test = results['y_test']
    y_pred = results['y_pred']
    
    st.subheader(f"📊 Model: {model_name}")
    
    # Metrics
    if problem_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f} ({accuracy*100:.2f}%)")
        with col2:
            st.metric("Precision", f"{precision:.4f} ({precision*100:.2f}%)")
        with col3:
            st.metric("Recall", f"{recall:.4f} ({recall*100:.2f}%)")
        with col4:
            st.metric("F1-Score", f"{f1:.4f} ({f1*100:.2f}%)")
        
        # Classification report
        st.subheader("📋 Classification Report")
        classes = results.get('classes', np.unique(y_test))
        report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4))
        
        # Confusion matrix
        st.subheader("🎯 Confusion Matrix")
        fig = create_confusion_matrix(y_test, y_pred, [str(c) for c in classes])
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(results['model'], 'feature_importances_'):
            st.subheader("🌟 Feature Importance")
            importance = results['model'].feature_importances_
            features = results['X_train'].columns if hasattr(results['X_train'], 'columns') else [f'Feature {i}' for i in range(len(importance))]
            fig = create_feature_importance_plot(features, importance)
            st.plotly_chart(fig, use_container_width=True)
        
        # Learning curve
        st.subheader("📈 Learning Curve")
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                results['model'], results['X_train'], results['y_train'],
                cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy', n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Training Accuracy',
                                   line=dict(color='blue'), error_y=dict(type='data', array=train_std)))
            fig.add_trace(go.Scatter(x=train_sizes, y=test_mean, mode='lines+markers', name='CV Accuracy',
                                   line=dict(color='green'), error_y=dict(type='data', array=test_std)))
            
            fig.update_layout(title='Learning Curve', xaxis_title='Training Examples', yaxis_title='Accuracy',
                            width=700, height=500)
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass
    
    else:  # Regression
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")
        with col4:
            st.metric("MSE", f"{mse:.4f}")
        
        # Residual plot
        st.subheader("📊 Residual Plot")
        residuals = y_test - y_pred
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                               marker=dict(size=8, color='blue', opacity=0.6)))
        fig.add_trace(go.Scatter(x=[y_pred.min(), y_pred.max()], y=[0, 0],
                               mode='lines', name='Zero Line', line=dict(color='red', dash='dash')))
        fig.update_layout(title='Residual Plot', xaxis_title='Predicted Values', yaxis_title='Residuals',
                         width=700, height=500)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
