import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
import io

# --- Page Configuration ---
st.set_page_config(page_title="GitHub ML Classifier", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #24292e; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #2ea44f; border: 1px solid #2ea44f; }
    .metric-card { background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    </style>
    """, unsafe_allow_html=True)

# --- Title & Header ---
st.imgae("PragyanAI_Transperent.png")
st.title("GitHub Developer Classification System")
st.markdown("""
This platform implements a complete **Machine Learning Lifecycle** to classify GitHub users into **Machine Learning** or **Web Development** categories using the *MUSAE GitHub Dataset*.
""")

# --- 1. Robust Data Ingestion ---
@st.cache_data
def load_and_merge_data():
    """Fetches real data from GitHub with a robust fallback mechanism."""
    # Current best URLs for MUSAE GitHub Dataset
    nodes_url = "https://raw.githubusercontent.com/benedekrozemberczki/datasets/master/musae_github_nodes.csv"
    targets_url = "https://raw.githubusercontent.com/benedekrozemberczki/datasets/master/musae_github_target.csv"
    
    try:
        # Set a timeout for the request to avoid hanging
        nodes_response = requests.get(nodes_url, timeout=10)
        targets_response = requests.get(targets_url, timeout=10)
        
        if nodes_response.status_code == 200 and targets_response.status_code == 200:
            nodes_df = pd.read_csv(io.StringIO(nodes_response.text))
            targets_df = pd.read_csv(io.StringIO(targets_response.text))
            
            # Merge datasets on the 'id' column
            df = pd.merge(nodes_df, targets_df, on='id')
            
            # Add synthetic features for classification (MUSAE lacks flat numeric features in node file)
            np.random.seed(42)
            df['repo_count'] = np.random.poisson(lam=25, size=len(df))
            df['stars_received'] = np.random.gamma(shape=2, scale=100, size=len(df))
            df['total_commits'] = df['repo_count'] * np.random.randint(50, 200, size=len(df))
            df['is_org_member'] = np.random.binomial(1, 0.4, size=len(df))
            df['developer_type'] = df['ml_target'].map({0: 'Web Developer', 1: 'ML Developer'})
            
            return df.sample(n=min(5000, len(df)), random_state=42)
        else:
            raise Exception("URL returned non-200 status code.")
            
    except Exception as e:
        # High-quality fallback if GitHub Raw is unreachable or 404s
        st.sidebar.warning(f"Using Fallback Data: {e}")
        np.random.seed(42)
        n = 2000
        data = {
            'id': np.arange(n),
            'repo_count': np.random.randint(5, 120, n),
            'stars_received': np.random.randint(0, 2000, n),
            'total_commits': np.random.randint(100, 10000, n),
            'is_org_member': np.random.choice([0, 1], n),
            'ml_target': np.random.choice([0, 1], n, p=[0.6, 0.4])
        }
        df = pd.DataFrame(data)
        df['developer_type'] = df['ml_target'].map({0: 'Web Developer', 1: 'ML Developer'})
        return df

df_raw = load_and_merge_data()

# --- Sidebar Controls ---
st.sidebar.image("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", width=60)
st.sidebar.header("ML Pipeline Steps")
app_mode = st.sidebar.selectbox("Choose a Phase", 
    ["1. Data Analysis", "2. Feature Engineering", "3. Model Training", "4. Performance & XAI", "5. Deployment Predictor"])

# Safety Check: If df_raw is somehow still None (shouldn't happen with fallback)
if df_raw is None:
    st.error("Fatal: Dataset could not be initialized. Please refresh.")
    st.stop()

# --- Phase 1: Data Analysis ---
if app_mode == "1. Data Analysis":
    st.header(" Exploratory Data Analysis (EDA)")
    
    col1, col2, col3 = st.columns(3)
    # Safely access length now
    col1.metric("Total Sample", f"{len(df_raw):,}")
    col2.metric("ML Developers", f"{len(df_raw[df_raw['ml_target']==1]):,}")
    col3.metric("Web Developers", f"{len(df_raw[df_raw['ml_target']==0]):,}")

    st.divider()
    
    tab1, tab2 = st.tabs(["Data Inspection", "Visual Analysis"])
    with tab1:
        st.subheader("Raw Dataset Preview")
        st.dataframe(df_raw.head(10), use_container_width=True)
        st.write("**Summary Information:**")
        buffer = io.StringIO()
        df_raw.info(buf=buffer)
        st.text(buffer.getvalue())
        
    with tab2:
        st.subheader("Feature Correlations")
        numeric_df = df_raw.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No numeric features available for correlation.")

# --- Phase 2: Feature Engineering ---
elif app_mode == "2. Feature Engineering":
    st.header(" Feature Engineering & Processing")
    
    st.markdown("""
    * **Feature Selection**: Using `repo_count`, `stars_received`, `total_commits`, and `is_org_member`.
    * **Normalization**: Applying `StandardScaler` for zero mean and unit variance.
    * **Data Splitting**: Preparing for supervised learning.
    """)
    
    features = ['repo_count', 'stars_received', 'total_commits', 'is_org_member']
    X = df_raw[features]
    y = df_raw['ml_target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    
    st.subheader("Engineered Feature Matrix")
    st.dataframe(X_scaled.head(10), use_container_width=True)
    
    st.session_state['processed'] = (X, y, features)
    st.success("Transformation complete! Session state updated.")

# --- Phase 3: Model Training ---
elif app_mode == "3. Model Training":
    st.header(" Model Selection & Tuning")
    
    if 'processed' not in st.session_state:
        st.error("Please run Phase 2 (Feature Engineering) first.")
    else:
        X, y, feat_names = st.session_state['processed']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            algo = st.radio("Select Algorithm", ["Random Forest", "Logistic Regression"])
            tune = st.checkbox("Enable Hyperparameter Tuning", value=True)
            
        if st.button("Start Training Pipeline"):
            with st.spinner(f"Optimizing {algo}..."):
                if algo == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                    params = {'n_estimators': [50, 100], 'max_depth': [5, 10]} if tune else {}
                else:
                    model = LogisticRegression(max_iter=1000)
                    params = {'C': [0.1, 1.0, 10.0]} if tune else {}
                
                # Perform Grid Search or Simple Fit
                search = GridSearchCV(model, params, cv=3) if tune else model.fit(X_train, y_train)
                if tune:
                    search.fit(X_train, y_train)
                    final_model = search.best_estimator_
                else:
                    final_model = search
                
                st.session_state['best_model'] = final_model
                st.session_state['algo_name'] = algo
                st.session_state['test_split'] = (X_test, y_test)
                
                st.success(f"Model Training Ready!")
                if tune: st.json(search.best_params_)

# --- Phase 4: Performance & XAI ---
elif app_mode == "4. Performance & XAI":
    st.header(" Evaluation & Explainability")
    
    if 'best_model' not in st.session_state:
        st.warning("Please train a model in Phase 3.")
    else:
        model = st.session_state['best_model']
        X_test, y_test = st.session_state['test_split']
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns([1, 3])
        acc = accuracy_score(y_test, y_pred)
        col1.metric("Accuracy Score", f"{acc*100:.2f}%")
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                           x=['Web Dev', 'ML Dev'], y=['Web Dev', 'ML Dev'])
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.divider()
        st.subheader(" Feature Importance (Global XAI)")
        if st.session_state['algo_name'] == "Random Forest":
            importances = model.feature_importances_
            feat_imp = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances}).sort_values('Importance')
            st.plotly_chart(px.bar(feat_imp, x='Importance', y='Feature', orientation='h'))
        else:
            weights = pd.DataFrame({'Feature': X_test.columns, 'Weight': model.coef_[0]}).sort_values('Weight')
            st.plotly_chart(px.bar(weights, x='Weight', y='Feature', orientation='h'))

# --- Phase 5: Deployment Predictor ---
elif app_mode == "5. Deployment Predictor":
    st.header(" Deployment Predictor")
    
    if 'best_model' not in st.session_state:
        st.error("Model unavailable. Please train a model first.")
    else:
        with st.form("predictor_form"):
            c1, c2 = st.columns(2)
            f_repos = c1.slider("Repos", 0, 200, 25)
            f_stars = c1.number_input("Stars", 0, 10000, 200)
            f_commits = c2.number_input("Commits", 0, 50000, 1000)
            f_org = c2.selectbox("Org Member", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            
            run_btn = st.form_submit_button("Classify User")
            
        if run_btn:
            input_df = pd.DataFrame([[f_repos, f_stars, f_commits, f_org]], 
                                   columns=['repo_count', 'stars_received', 'total_commits', 'is_org_member'])
            model = st.session_state['best_model']
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][pred]
            
            label = "ML Developer" if pred == 1 else "Web Developer"
            st.subheader(f"Prediction: {label}")
            st.progress(float(prob))
            st.write(f"Model Confidence: {prob*100:.2f}%")
