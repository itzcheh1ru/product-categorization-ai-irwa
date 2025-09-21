'''import streamlit as st
import requests
import pandas as pd
import json
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Product Categorization AI",
    page_icon="🛍️",
    layout="wide"
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
    .result-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .tag {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

def call_api(endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Helper function to call API endpoints"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if data:
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def main():
    st.markdown('<h1 class="main-header">🛍️ Product Categorization AI</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a mode",
        ["Single Product Processing", "Batch Processing", "API Health Check"]
    )
    
    if app_mode == "Single Product Processing":
        single_product_processing()
    elif app_mode == "Batch Processing":
        batch_processing()
    elif app_mode == "API Health Check":
        health_check()

def single_product_processing():
    st.header("Process Single Product")
    
    # Product input form
    with st.form("product_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_name = st.text_input("Product Name*", placeholder="e.g., Nike Men Running Shoes")
            gender = st.selectbox("Gender", ["", "Men", "Women", "Boys", "Girls", "Unisex"])
            base_color = st.text_input("Base Color", placeholder="e.g., Blue")
            season = st.selectbox("Season", ["", "Summer", "Winter", "Spring", "Fall", "All Season"])
        
        with col2:
            category = st.selectbox("Category", ["", "Apparel", "Accessories", "Footwear", "Personal Care", "Free Items"])
            usage = st.text_input("Usage", placeholder="e.g., Casual, Sports")
            year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)
        
        submitted = st.form_submit_button("Process Product")
    
    if submitted and product_name:
        # Prepare product data
        product_data = {
            "productDisplayName": product_name,
            "gender": gender,
            "masterCategory": category,
            "baseColour": base_color,
            "season": season,
            "year": year,
            "usage": usage
        }
        
        # Call API
        with st.spinner("Processing product..."):
            result = call_api("/api/orchestrator/process", product_data)
        
        if result:
            display_results(result)

def display_results(result: Dict[str, Any]):
    st.success("Product processed successfully!")
    
    # Display classification results
    st.subheader("Classification Results")
    classification = result.get('classification', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Category", classification.get('category', 'Unknown'))
    with col2:
        st.metric("Subcategory", classification.get('subcategory', 'Unknown'))
    with col3:
        confidence = classification.get('confidence', 0)
        st.metric("Confidence", f"{confidence:.2%}")
    
    if classification.get('reasoning'):
        st.info(f"Reasoning: {classification.get('reasoning')}")
    
    # Display attributes
    st.subheader("Extracted Attributes")
    attributes = result.get('attributes', {}).get('attributes', {})
    
    attr_cols = st.columns(2)
    for i, (attr_name, attr_value) in enumerate(attributes.items()):
        if isinstance(attr_value, dict) and 'value' in attr_value:
            col_idx = i % 2
            with attr_cols[col_idx]:
                confidence = attr_value.get('confidence', 0)
                st.metric(
                    label=attr_name.title(),
                    value=attr_value.get('value', 'Unknown'),
                    delta=f"{confidence:.0%} confidence" if confidence > 0 else None
                )
    
    # Display tags
    st.subheader("Generated Tags")
    tags = result.get('tags', {}).get('tags', [])
    
    tag_html = "".join([f'<span class="tag">{tag["tag"]} ({tag.get("confidence", 0):.0%})</span>' 
                       for tag in tags[:10]])
    st.markdown(f'<div>{tag_html}</div>', unsafe_allow_html=True)
    
    # Display similar products
    similar_products = result.get('similar_products', [])
    if similar_products:
        st.subheader("Similar Products")
        for product in similar_products:
            with st.expander(f"{product.get('productDisplayName', 'Unknown')} ({product.get('similarity_score', 0):.2%} similar)"):
                st.write(f"Category: {product.get('masterCategory', 'Unknown')}")
                st.write(f"Type: {product.get('articleType', 'Unknown')}")
                st.write(f"Color: {product.get('baseColour', 'Unknown')}")

def batch_processing():
    st.header("Batch Process Products")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        if st.button("Process Batch"):
            results = []
            progress_bar = st.progress(0)
            
            for i, row in df.iterrows():
                product_data = row.to_dict()
                result = call_api("/api/orchestrator/process", product_data)
                
                if result:
                    results.append({
                        "product": product_data.get('productDisplayName', ''),
                        "category": result.get('classification', {}).get('category', ''),
                        "confidence": result.get('classification', {}).get('confidence', 0),
                        "tags": [tag['tag'] for tag in result.get('tags', {}).get('tags', [])][:5]
                    })
                
                progress_bar.progress((i + 1) / len(df))
            
            # Display results
            results_df = pd.DataFrame(results)
            st.subheader("Batch Processing Results")
            st.dataframe(results_df)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="batch_processing_results.csv",
                mime="text/csv"
            )

def health_check():
    st.header("API Health Check")
    
    if st.button("Check API Status"):
        result = call_api("/health")
        
        if result:
            st.success("API is healthy! 🟢")
            st.json(result)
        else:
            st.error("API is not responding 🔴")

if __name__ == "__main__":
    main()
    '''