import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings in the Streamlit app
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍️",
    layout="wide"
)

# --- Caching Function for Data Loading ---
# This improves performance by caching the data
@st.cache_data
def load_data(file_path):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the same directory as the app.py file.")
        return None

# --- Plotting Functions ---
# We create separate functions for plots to keep the code organized.

def plot_elbow_method(data):
    """Plots the elbow curve to find the optimal k."""
    inertia_scores = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        inertia_scores.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 11), inertia_scores, marker='o', linestyle='--')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    st.pyplot(fig)

def plot_bivariate_clusters(df, x_col, y_col, cluster_col, centers):
    """Plots the bivariate clustering results."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=cluster_col, palette='tab10', ax=ax, s=60)
    
    # Plotting the cluster centers
    ax.scatter(x=centers[:, 0], y=centers[:, 1], s=200, c='black', marker='*', label='Centroids')
    
    ax.set_title(f'Customer Segments based on {x_col} and {y_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(title='Cluster')
    st.pyplot(fig)

# --- Main Application Logic ---
def main():
    st.title("🛍️ Customer Segmentation and Clustering")
    st.markdown("An interactive web app to analyze and cluster customer data.")

    # --- Sidebar for Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Data Analysis", "Clustering Analysis"])

    # Load the data
    df_original = load_data("Mall_Customers.csv")

    if df_original is None:
        return # Stop execution if data loading failed

    # --- Data Overview Page ---
    if page == "Data Overview":
        st.header("1. Data Overview")
        st.write("Here is a preview of the dataset:")
        st.dataframe(df_original.head())
        
        st.header("2. Dataset Statistics")
        st.write("A summary of the numerical features in the dataset:")
        st.write(df_original.describe())

    # --- EDA Page ---
    elif page == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis (EDA)")
        st.markdown("Visualize the data to understand its distribution and relationships.")

        # Let user select a plot type
        plot_type = st.selectbox(
            "Select a type of visualization",
            ["Distribution Plot", "Box Plot", "Scatter Plot", "Pair Plot", "Correlation Heatmap"]
        )

        # Columns for plotting
        columns_to_plot = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        
        if plot_type == "Distribution Plot":
            selected_col = st.selectbox("Select a feature to visualize its distribution", columns_to_plot)
            fig, ax = plt.subplots()
            sns.distplot(df_original[selected_col], ax=ax)
            st.pyplot(fig)

        elif plot_type == "Box Plot":
            selected_col = st.selectbox("Select a feature to see its relationship with Gender", columns_to_plot)
            fig, ax = plt.subplots()
            sns.boxplot(data=df_original, x='Gender', y=selected_col, ax=ax)
            st.pyplot(fig)

        elif plot_type == "Scatter Plot":
            st.subheader("Annual Income vs. Spending Score")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_original, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', ax=ax)
            st.pyplot(fig)

        elif plot_type == "Pair Plot":
            st.subheader("Pairwise relationships between features")
            # Drop CustomerID for a cleaner pairplot
            df_for_pairplot = df_original.drop('CustomerID', axis=1)
            fig = sns.pairplot(df_for_pairplot, hue='Gender')
            st.pyplot(fig)

        elif plot_type == "Correlation Heatmap":
            st.subheader("Correlation Matrix of Features")
            # Drop CustomerID as it's not a feature
            corr_df = df_original.drop('CustomerID', axis=1).corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    # --- Clustering Page ---
    elif page == "Clustering Analysis":
        st.header("Customer Segmentation using K-Means Clustering")

        # Create a copy for clustering to avoid modifying the original df
        df_cluster = df_original.copy()
        
        cluster_type = st.selectbox(
            "Select the features for clustering",
            ["Bivariate (Income & Spending Score)", "Multivariate (Age, Income, Spending Score, Gender)"]
        )

        if cluster_type == "Bivariate (Income & Spending Score)":
            st.subheader("1. Finding the Optimal Number of Clusters (k)")
            st.markdown("The **Elbow Method** helps us find the best `k` by looking for the 'elbow' in the plot, which indicates the point of diminishing returns.")
            
            bivariate_data = df_cluster[['Annual Income (k$)', 'Spending Score (1-100)']]
            plot_elbow_method(bivariate_data)

            st.subheader("2. Perform Bivariate Clustering")
            k_bivariate = st.slider("Select the number of clusters (k)", min_value=2, max_value=10, value=5, key="bivariate")
            
            kmeans_biv = KMeans(n_clusters=k_bivariate, random_state=42)
            df_cluster['Cluster'] = kmeans_biv.fit_predict(bivariate_data)
            
            st.markdown(f"#### Visualization of the {k_bivariate} Clusters")
            plot_bivariate_clusters(
                df_cluster, 
                'Annual Income (k$)', 
                'Spending Score (1-100)', 
                'Cluster', 
                kmeans_biv.cluster_centers_
            )

            st.markdown("#### Cluster Profiles")
            st.write("Let's examine the average characteristics of each customer segment:")
            cluster_means = df_cluster.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
            st.dataframe(cluster_means)

        elif cluster_type == "Multivariate (Age, Income, Spending Score, Gender)":
            st.subheader("1. Data Preprocessing")
            st.markdown("""
            Before clustering with multiple features, we need to:
            1.  **Encode Categorical Data**: Convert 'Gender' into a numerical format (0s and 1s).
            2.  **Scale Numerical Data**: Use `StandardScaler` to ensure all features have a similar scale, preventing features with larger values (like Annual Income) from dominating the clustering process.
            """)
            
            # Preprocessing
            df_multi = df_cluster.drop('CustomerID', axis=1)
            df_multi = pd.get_dummies(df_multi, columns=['Gender'], drop_first=True)
            
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_multi)
            df_scaled = pd.DataFrame(scaled_features, columns=df_multi.columns)
            
            st.subheader("2. Finding the Optimal Number of Clusters (k)")
            plot_elbow_method(df_scaled)
            
            st.subheader("3. Perform Multivariate Clustering")
            k_multivariate = st.slider("Select the number of clusters (k)", min_value=2, max_value=10, value=6, key="multivariate")

            kmeans_multi = KMeans(n_clusters=k_multivariate, random_state=42)
            df_cluster['Cluster'] = kmeans_multi.fit_predict(df_scaled)
            
            st.markdown("#### Cluster Profiles")
            st.write("Since we cannot easily visualize 4D data, we analyze the segments by looking at their average characteristics:")
            cluster_means_multi = df_cluster.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
            st.dataframe(cluster_means_multi)
            
            st.write("Gender distribution within clusters:")
            gender_dist = pd.crosstab(df_cluster['Cluster'], df_cluster['Gender'])
            st.dataframe(gender_dist)

        # Add a download button for the clustered data
        st.sidebar.markdown("---")
        st.sidebar.header("Download Results")
        
        # Function to convert DataFrame to CSV for download
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(df_cluster)
        st.sidebar.download_button(
            label="Download Data with Clusters",
            data=csv,
            file_name='clustered_customers.csv',
            mime='text/csv',
        )


if __name__ == '__main__':
    main()