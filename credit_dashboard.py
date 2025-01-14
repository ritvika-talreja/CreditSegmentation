import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# Load your dataset
data = pd.read_csv("Credit_Score.csv")

# Map categorical features for preprocessing
education_level_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
employment_status_mapping = {'Unemployed': 0, 'Employed': 1, 'Self-Employed': 2}

data['Education Level'] = data['Education Level'].map(education_level_mapping)
data['Employment Status'] = data['Employment Status'].map(employment_status_mapping)

# Handle missing values
data.fillna(0, inplace=True)

# Calculate credit scores (FICO-like formula)
data['Credit Score'] = (
    data['Payment History'] * 0.35 +
    data['Credit Utilization Ratio'] * 0.30 +
    data['Number of Credit Accounts'] * 0.15 +
    data['Education Level'] * 0.10 +
    data['Employment Status'] * 0.10
)

# Segmentation using KMeans
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
data['Segment'] = kmeans.fit_predict(data[['Credit Score']])
data['Segment'] = data['Segment'].map({2: 'Very Low', 0: 'Low', 1: 'Good', 3: 'Excellent'})

# Credit score ranges for cohort analysis
bins = [300, 500, 700, 850, 900]  # Adjusted bin edges
labels = ['Poor', 'Fair', 'Good', 'Excellent']
data['Credit Score Range'] = pd.cut(data['Credit Score'], bins=bins, labels=labels, right=False)

# Simulate time periods (if not in dataset)
data['Time Period'] = np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], size=len(data))

# Streamlit Dashboard
st.title("Credit Scoring and Segmentation Dashboard")

# Tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Segment Analysis", "Cohort Analysis", "Feature Importance", "Clusters"])

# Tab 1: Overview
with tab1:
    st.header("Dataset Overview")
    st.write("### Sample Data")
    st.dataframe(data.head())

# Tab 2: Segment Analysis
with tab2:
    st.header("Segment Analysis")
    segment = st.selectbox("Select a Segment:", sorted(data['Segment'].unique()))
    filtered_data = data[data['Segment'] == segment]

    # Scatter Plot
    scatter_fig = px.scatter(
        filtered_data, 
        x=filtered_data.index, 
        y='Credit Score', 
        color='Segment',
        title=f'Credit Scores in {segment} Segment'
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Feature Comparison
    st.subheader(f"Feature Comparison for {segment} Segment")
    features = ['Credit Utilization Ratio', 'Loan Amount', 'Payment History']
    numeric_columns = data.select_dtypes(include=['number']).columns
    grouped_data = data.groupby('Segment')[numeric_columns].mean()
    bar_fig = px.bar(
        grouped_data.loc[[segment], features].transpose(),
        title=f'Average Feature Values in {segment} Segment',
        labels={'value': 'Average Value', 'index': 'Feature'}
    )
    st.plotly_chart(bar_fig, use_container_width=True)

# Tab 3: Cohort Analysis
with tab3:
    st.header("Cohort Analysis")
    cohort_data = data.groupby(['Time Period', 'Credit Score Range']).size().reset_index(name='Count')
    cohort_fig = px.bar(
        cohort_data, 
        x='Time Period', 
        y='Count', 
        color='Credit Score Range',
        title='Credit Score Cohorts Over Time', 
        barmode='stack'
    )
    st.plotly_chart(cohort_fig, use_container_width=True)

# Tab 4: Feature Importance
with tab4:
    st.header("Feature Importance Analysis")
    # Train a Random Forest model
    rf_model = RandomForestRegressor()
    X = data[['Credit Utilization Ratio', 'Loan Amount', 'Payment History', 'Number of Credit Accounts', 'Interest Rate', 'Loan Term']]
    y = data['Credit Score']
    rf_model.fit(X, y)

    # Feature Importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    pie_fig = px.pie(
        feature_importance, 
        names='Feature', 
        values='Importance', 
        title='Feature Importance Proportion'
    )
    st.plotly_chart(pie_fig, use_container_width=True)

# Tab 5: Clusters
with tab5:
    st.header("Cluster Visualization")
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    data['PCA1'] = pca_result[:, 0]
    data['PCA2'] = pca_result[:, 1]

    # Scatter Plot for Clusters
    cluster_fig = px.scatter(
        data, 
        x='PCA1', 
        y='PCA2', 
        color='Segment',
        title='Clusters in 2D PCA Space',
        labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'}
    )
    st.plotly_chart(cluster_fig, use_container_width=True)

# Recommendations
st.sidebar.header("Customer Recommendations")
recommendations = {
    'Very Low': 'Focus on improving payment history and reducing credit utilization ratio.',
    'Low': 'Work on increasing the number of credit accounts responsibly.',
    'Good': 'Maintain current habits and avoid high credit utilization.',
    'Excellent': 'Consider opportunities for additional financial products.'
}
selected_segment = st.sidebar.selectbox("Choose a Segment to Get Recommendations:", data['Segment'].unique())
st.sidebar.write(recommendations[selected_segment])
