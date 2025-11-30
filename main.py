import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import io

# Page configuration
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class DataAnalysisApp:
    def __init__(self):
        self.df = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "Data Explorer"
    
    def main(self):
        """Main application function"""
        # Header
        st.markdown('<h1 class="main-header">📊 Data Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content based on selected tab
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Data Explorer", 
            "📈 Visualization", 
            "🤖 ML Modeling", 
            "📋 Statistics", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self.data_explorer_tab()
        
        with tab2:
            self.visualization_tab()
        
        with tab3:
            self.ml_modeling_tab()
        
        with tab4:
            self.statistics_tab()
        
        with tab5:
            self.settings_tab()
    
    def render_sidebar(self):
        """Render sidebar components"""
        st.sidebar.title("Navigation")
        
        # Data upload section
        st.sidebar.header("📁 Data Management")
        
        upload_option = st.sidebar.radio(
            "Choose data source:",
            ["Upload CSV", "Use Sample Data"]
        )
        
        if upload_option == "Upload CSV":
            uploaded_file = st.sidebar.file_uploader(
                "Upload your CSV file", 
                type=['csv'],
                help="Upload a CSV file to analyze"
            )
            
            if uploaded_file is not None:
                try:
                    self.df = pd.read_csv(uploaded_file)
                    st.session_state.data_loaded = True
                    st.sidebar.success(f"✅ Data loaded successfully! Shape: {self.df.shape}")
                except Exception as e:
                    st.sidebar.error(f"Error loading file: {str(e)}")
        
        else:  # Use Sample Data
            sample_option = st.sidebar.selectbox(
                "Choose sample dataset:",
                ["Iris Classification", "Sales Data", "Random Regression"]
            )
            
            if st.sidebar.button("Generate Sample Data"):
                self.generate_sample_data(sample_option)
        
        # Data info if loaded
        if st.session_state.data_loaded and self.df is not None:
            st.sidebar.header("📊 Data Info")
            st.sidebar.write(f"**Shape:** {self.df.shape}")
            st.sidebar.write(f"**Columns:** {len(self.df.columns)}")
            st.sidebar.write(f"**Memory usage:** {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    def generate_sample_data(self, dataset_type):
        """Generate sample datasets"""
        if dataset_type == "Iris Classification":
            from sklearn.datasets import load_iris
            iris = load_iris()
            self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.df['target'] = iris.target
            self.df['species'] = self.df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        elif dataset_type == "Sales Data":
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=1000, freq='D')
            self.df = pd.DataFrame({
                'date': dates,
                'product': np.random.choice(['Product A', 'Product B', 'Product C'], 1000),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
                'sales': np.random.normal(1000, 200, 1000),
                'quantity': np.random.randint(1, 50, 1000),
                'customer_rating': np.random.uniform(1, 5, 1000)
            })
            self.df['revenue'] = self.df['sales'] * self.df['quantity']
        
        else:  # Random Regression
            X, y = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)
            self.df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(4)])
            self.df['target'] = y
        
        st.session_state.data_loaded = True
        st.sidebar.success(f"✅ {dataset_type} sample data generated! Shape: {self.df.shape}")
    
    def data_explorer_tab(self):
        """Data explorer tab content"""
        st.header("🔍 Data Explorer")
        
        if not st.session_state.data_loaded:
            st.info("👆 Please upload a CSV file or generate sample data from the sidebar to get started!")
            return
        
        # Data preview
        st.subheader("Data Preview")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            show_rows = st.slider("Number of rows to show", 5, 100, 10)
        with col2:
            start_row = st.number_input("Start from row", 0, len(self.df)-1, 0)
        with col3:
            st.metric("Total Rows", len(self.df))
        
        st.dataframe(self.df.iloc[start_row:start_row + show_rows], use_container_width=True)
        
        # Data information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame(self.df.dtypes, columns=['Data Type'])
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Missing Values")
            missing_df = pd.DataFrame(self.df.isnull().sum(), columns=['Missing Values'])
            missing_df['Percentage'] = (missing_df['Missing Values'] / len(self.df)) * 100
            st.dataframe(missing_df, use_container_width=True)
        
        # Column operations
        st.subheader("Column Operations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Show Column Names"):
                st.write(list(self.df.columns))
        
        with col2:
            if st.button("Describe Data"):
                st.dataframe(self.df.describe(), use_container_width=True)
        
        with col3:
            if st.download_button(
                label="Download Data as CSV",
                data=self.df.to_csv(index=False),
                file_name="analyzed_data.csv",
                mime="text/csv"
            ):
                st.success("Data downloaded successfully!")
    
    def visualization_tab(self):
        """Visualization tab content"""
        st.header("📈 Data Visualization")
        
        if not st.session_state.data_loaded:
            st.info("👆 Please load data first in the Data Explorer tab!")
            return
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # X-axis selection
            x_axis = st.selectbox("X-axis", self.df.select_dtypes(include=[np.number]).columns.tolist())
        
        with col2:
            # Y-axis selection (for relevant plots)
            if viz_type in ["Scatter Plot", "Line Chart"]:
                y_axis = st.selectbox("Y-axis", self.df.select_dtypes(include=[np.number]).columns.tolist())
        
        # Additional options based on visualization type
        if viz_type == "Scatter Plot":
            color_by = st.selectbox("Color by", [None] + self.df.columns.tolist())
            fig = px.scatter(self.df, x=x_axis, y=y_axis, color=color_by, 
                           title=f"Scatter Plot: {x_axis} vs {y_axis}")
        
        elif viz_type == "Line Chart":
            fig = px.line(self.df, x=x_axis, y=y_axis, title=f"Line Chart: {x_axis} vs {y_axis}")
        
        elif viz_type == "Bar Chart":
            y_axis = st.selectbox("Y-axis", self.df.select_dtypes(include=[np.number]).columns.tolist())
            fig = px.bar(self.df, x=x_axis, y=y_axis, title=f"Bar Chart: {x_axis} vs {y_axis}")
        
        elif viz_type == "Histogram":
            fig = px.histogram(self.df, x=x_axis, title=f"Histogram of {x_axis}")
        
        elif viz_type == "Box Plot":
            y_axis = st.selectbox("Y-axis", self.df.select_dtypes(include=[np.number]).columns.tolist())
            fig = px.box(self.df, x=x_axis, y=y_axis, title=f"Box Plot: {x_axis} vs {y_axis}")
        
        else:  # Heatmap
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            corr_matrix = self.df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title="Correlation Heatmap", aspect="auto")
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualization controls
        st.subheader("Visualization Controls")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Update Chart"):
                st.rerun()
        
        with col2:
            if st.button("Save Chart as HTML"):
                fig.write_html("chart.html")
                st.success("Chart saved as chart.html")
    
    def ml_modeling_tab(self):
        """Machine Learning modeling tab"""
        st.header("🤖 Machine Learning Modeling")
        
        if not st.session_state.data_loaded:
            st.info("👆 Please load data first in the Data Explorer tab!")
            return
        
        # Model type selection
        model_type = st.radio(
            "Select model type:",
            ["Classification", "Regression"]
        )
        
        # Feature and target selection
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for modeling!")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox("Target variable", numeric_cols)
        
        with col2:
            feature_cols = st.multiselect(
                "Feature variables",
                [col for col in numeric_cols if col != target_col],
                default=[col for col in numeric_cols if col != target_col][:3]
            )
        
        if not feature_cols:
            st.warning("Please select at least one feature variable!")
            return
        
        # Model parameters
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
            n_estimators = st.slider("Number of estimators", 10, 200, 100)
        
        with col2:
            random_state = st.number_input("Random state", 0, 100, 42)
            max_depth = st.slider("Max depth", 1, 20, 10)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Prepare data
                    X = self.df[feature_cols]
                    y = self.df[target_col]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Train model
                    if model_type == "Classification":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Display results
                        st.success(f"✅ Model trained successfully!")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Accuracy", f"{accuracy:.3f}")
                        col2.metric("Training samples", len(X_train))
                        col3.metric("Test samples", len(X_test))
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame({
                            'feature': feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=True)
                        
                        fig = px.bar(importance_df, x='importance', y='feature', 
                                   orientation='h', title="Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # Regression
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        
                        # Display results
                        st.success(f"✅ Model trained successfully!")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("RMSE", f"{rmse:.3f}")
                        col2.metric("Training samples", len(X_train))
                        col3.metric("Test samples", len(X_test))
                        
                        # Prediction vs Actual plot
                        fig = px.scatter(x=y_test, y=y_pred, 
                                       labels={'x': 'Actual', 'y': 'Predicted'},
                                       title="Actual vs Predicted Values")
                        fig.add_shape(type='line', line=dict(dash='dash'),
                                    x0=y_test.min(), y0=y_test.min(),
                                    x1=y_test.max(), y1=y_test.max())
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    
    def statistics_tab(self):
        """Statistical analysis tab"""
        st.header("📋 Statistical Analysis")
        
        if not st.session_state.data_loaded:
            st.info("👆 Please load data first in the Data Explorer tab!")
            return
        
        # Basic statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(self.df.describe(), use_container_width=True)
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title="Correlation Matrix", 
                          color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        # Outlier detection
        st.subheader("Outlier Analysis")
        numeric_col = st.selectbox("Select column for outlier analysis", numeric_cols)
        
        if numeric_col:
            Q1 = self.df[numeric_col].quantile(0.25)
            Q3 = self.df[numeric_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[numeric_col] < lower_bound) | (self.df[numeric_col] > upper_bound)]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lower Bound", f"{lower_bound:.2f}")
            col2.metric("Upper Bound", f"{upper_bound:.2f}")
            col3.metric("Number of Outliers", len(outliers))
            col4.metric("Outlier Percentage", f"{(len(outliers)/len(self.df))*100:.2f}%")
            
            if len(outliers) > 0:
                st.write("Outlier samples:")
                st.dataframe(outliers.head(10), use_container_width=True)
    
    def settings_tab(self):
        """Settings tab"""
        st.header("⚙️ Application Settings")
        
        st.subheader("Data Management")
        
        if st.session_state.data_loaded and self.df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear Current Data"):
                    st.session_state.data_loaded = False
                    self.df = None
                    st.rerun()
            
            with col2:
                if st.button("Reset All Settings"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
        
        st.subheader("About")
        st.markdown("""
        This Data Analysis Dashboard is built with Streamlit and provides:
        - 📊 Data exploration and visualization
        - 🤖 Machine learning modeling
        - 📋 Statistical analysis
        - ⚙️ Customizable settings
        
        **Libraries used:** Streamlit, Pandas, Plotly, Scikit-learn, NumPy
        """)

# Run the application
if __name__ == "__main__":
    app = DataAnalysisApp()
    app.main()
