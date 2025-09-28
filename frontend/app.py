import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import ast
import ollama  # Ensure you have the ollama package installed and configured
from src.utils.api import process_product_description, get_recommended_products

# Configuration
API_BASE_URL = "http://localhost:8000/api"  # Update with your FastAPI server URL

# Resolve data path dynamically (prefer product.csv, fall back to cleaned_product_data.csv)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "backend", "data", "product.csv"),
    os.path.join(PROJECT_ROOT, "backend", "data", "cleaned_product_data.csv"),
]
SELECTED_DATA_PATH = None

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'customer_home'
if 'customer_page' not in st.session_state:
    st.session_state.customer_page = 'home'
if 'admin_page' not in st.session_state:
    st.session_state.admin_page = 'dashboard'
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'selected_subcategory' not in st.session_state:
    st.session_state.selected_subcategory = None
if 'selected_article_type' not in st.session_state:
    st.session_state.selected_article_type = None
if 'selected_tag' not in st.session_state:
    st.session_state.selected_tag = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'product_data' not in st.session_state:
    st.session_state.product_data = None
if 'admin_data' not in st.session_state:
    st.session_state.admin_data = None
if 'agent_results' not in st.session_state:
    st.session_state.agent_results = {}
if 'relevant_products' not in st.session_state:
    st.session_state.relevant_products = None


def load_product_data():
    """Load product data from CSV file with the correct structure"""
    try:
        # pick the first existing path from candidates
        data_path = None
        for candidate in DATA_PATH_CANDIDATES:
            if os.path.exists(candidate):
                data_path = candidate
                break

        if data_path and os.path.exists(data_path):
            # store selected path for display
            global SELECTED_DATA_PATH
            SELECTED_DATA_PATH = data_path
            df = pd.read_csv(data_path)
            
            # Create tags from available attributes for better discovery
            df['tags'] = df.apply(lambda row: [
                row['gender'],
                row['masterCategory'],
                row['subCategory'],
                row['articleType'],
                row['baseColour'],
                row['season'],
                str(int(row['year'])) if pd.notna(row['year']) else '',
                row['usage']
            ], axis=1)
            
            # Clean up tags - remove empty values and flatten
            df['tags'] = df['tags'].apply(lambda x: [tag for tag in x if tag and str(tag).strip() != ''])
            
            st.session_state.admin_data = df
            return df
        else:
            st.warning("Product data file not found locally. Using sample data.")
            return load_sample_data()
    except Exception as e:
        st.error(f"Error loading product data: {str(e)}")
        return load_sample_data()

def load_sample_data():
    """Load sample product data for demonstration if real data is unavailable"""
    sample_data = [
        {
            "id": 15970,
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "Shirts",
            "baseColour": "Navy Blue",
            "season": "Autumn",
            "year": 2011.0,
            "usage": "Casual",
            "productDisplayName": "Turtle Check Men Navy Blue Shirt",
            "filename": "15970.jpg",
            "link": "http://assets.myntassets.com/v1/images/style/properties/7a5b82d1372a7a5c6de67ae7a314fd91_images.jpg"
        },
        {
            "id": 39386,
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Bottomwear",
            "articleType": "Jeans",
            "baseColour": "Blue",
            "season": "Summer",
            "year": 2012.0,
            "usage": "Casual",
            "productDisplayName": "Peter England Men Party Blue Jeans",
            "filename": "39386.jpg",
            "link": "http://assets.myntassets.com/v1/images/style/properties/4850873d0c417e6480a26059f83aac29_images.jpg"
        },
        {
            "id": 59263,
            "gender": "Women",
            "masterCategory": "Accessories",
            "subCategory": "Watches",
            "articleType": "Watches",
            "baseColour": "Silver",
            "season": "Winter",
            "year": 2016.0,
            "usage": "Casual",
            "productDisplayName": "Titan Women Silver Watch",
            "filename": "59263.jpg",
            "link": "http://assets.myntassets.com/v1/images/style/properties/Titan-Women-Silver-Watch_b4ef04538840c0020e4829ecc042ead1_images.jpg"
        }
    ]
    df = pd.DataFrame(sample_data)
    df['tags'] = df.apply(lambda row: [
        row['gender'],
        row['masterCategory'],
        row['subCategory'],
        row['articleType'],
        row['baseColour'],
        row['season'],
        str(int(row['year'])) if pd.notna(row['year']) else '',
        row['usage']
    ], axis=1)
    return df

def call_api(endpoint, data=None, method="POST"):
    """Generic API call function"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if method == "POST":
            response = requests.post(url, json=data, headers=headers)
        else:
            response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def process_product_description(description):
    """Process product description through the orchestrator"""
    product_data = {"description": description}
    result = call_api("/orchestrator/process", product_data)
    return result

def get_categories():
    """Get available categories from the actual data"""
    df = load_product_data()
    if df is not None and 'masterCategory' in df.columns:
        return sorted(df['masterCategory'].unique().tolist())
    return ["Apparel", "Accessories", "Footwear"]

def get_subcategories(category=None):
    """Get available subcategories"""
    df = load_product_data()
    if df is not None and 'subCategory' in df.columns:
        if category:
            return sorted(df[df['masterCategory'] == category]['subCategory'].unique().tolist())
        return sorted(df['subCategory'].unique().tolist())
    return ["Topwear", "Bottomwear", "Watches"]

def get_article_types(subcategory=None):
    """Get available article types"""
    df = load_product_data()
    if df is not None and 'articleType' in df.columns:
        if subcategory:
            return sorted(df[df['subCategory'] == subcategory]['articleType'].unique().tolist())
        return sorted(df['articleType'].unique().tolist())
    return ["Shirts", "Jeans", "Watches"]

def display_products(df, category=None, subcategory=None, article_type=None, tag=None):
    """Display products in a grid layout like an e-commerce shop page"""
    if df is None or df.empty:
        st.info("No products found.")
        return

    # Apply filters
    if category:
        df = df[df['masterCategory'] == category]
    if subcategory:
        df = df[df['subCategory'] == subcategory]
    if article_type:
        df = df[df['articleType'] == article_type]
    if tag:
        df = df[df['tags'].apply(lambda x: tag in x if isinstance(x, list) else False)]

    if df.empty:
        st.info("No products found with the selected criteria.")
        return

    # Show in grid (3 per row)
    num_cols = 3
    for i in range(0, len(df), num_cols):
        cols = st.columns(num_cols)
        for j, product in enumerate(df.iloc[i:i+num_cols].iterrows()):
            _, prod = product
            with cols[j]:
                # Product image
                if 'link' in prod and isinstance(prod['link'], str) and prod['link'].strip():
                    st.image(prod['link'], width=200)
                else:
                    st.image("https://via.placeholder.com/300x400?text=No+Image", use_container_width=True)

                # Product title
                st.markdown(f"**{prod.get('productDisplayName', 'Unnamed Product')}**")

                # Price (mock price if missing)
                price = prod.get("price", None)
                if price:
                    st.write(f"💰 LKR {price:,.2f}")
                else:
                    st.write("💰 Price on request")

                 # Tags (inside grid card)
                if 'tags' in prod and isinstance(prod['tags'], list) and len(prod['tags']) > 0:
                    tag_buttons = []
                    for i, t in enumerate(prod['tags']):
                        key = f"tag_{prod['id']}_{i}"
                        if st.button(f"#{t}", key=key):
                            st.session_state.selected_tag = t
                            st.session_state.customer_page = 'tag_products'
                            st.rerun()

                # View button
                if st.button("View Details", key=f"view_{prod['id']}"):
                    st.session_state.selected_product = prod.to_dict()
                    st.session_state.customer_page = 'product_detail'
                    st.rerun()


# -----------------
# Ollama summary
# -----------------
def generate_summary(description: str) -> str:
    try:
        response = ollama.chat(
            model="gemma3:270m",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize this shopping request in 1-2 short sentences: {description}"}
            ]
        )
        return response.get("message", {}).get("content") or response.get("content") or "⚠️ Could not parse summary."
    except Exception as e:
        return f"⚠️ Could not generate summary ({e})"


# ==================== CUSTOMER PAGES ====================

def customer_home_page():
    """Customer home page"""
    st.title("🛍️ Welcome to Product AI")
    st.write("Discover the perfect products with AI-powered search and recommendations!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 AI Search", use_container_width=True, help="Describe what you're looking for"):
            st.session_state.customer_page = 'ai_search'
            st.rerun()
    
    with col2:
        if st.button("📂 Browse Categories", use_container_width=True, help="Browse by product categories"):
            st.session_state.customer_page = 'browse'
            st.rerun()
    
    with col3:
        if st.button("🏷️ Popular Tags", use_container_width=True, help="Browse by popular tags"):
            st.session_state.customer_page = 'tags'
            st.rerun()
    
    st.markdown("---")
    st.subheader("🎯 Featured Products")
    
    df = load_product_data()
    if df is not None and not df.empty:
        # Show random sample of products
        sample_products = df.sample(min(6, len(df)))
        display_products(sample_products)
    else:
        st.info("No products available at the moment.")

def customer_ai_search_page():
    st.title("🔍 AI Product Search")

    if st.button("← Back to Home"):
        st.session_state.customer_page = 'home'
        st.rerun()

    description = st.text_area(
        "Describe what you're looking for:",
        placeholder="e.g., 'I need comfortable blue jeans for casual wear'",
        height=100
    )

    if st.button("Find Products", type="primary"):
        if not description:
            st.warning("Please enter a product description.")
            return

        # Clear any previous search results
        if 'search_results' in st.session_state:
            del st.session_state.search_results

        with st.spinner("Analyzing your description and finding matching products..."):
            # Step 1: Generate summary using Ollama
            try:
                summary_text = generate_summary(description)
            except Exception:
                summary_text = description  # fallback

            # Optional backend processing
            _ = process_product_description(description)

            st.session_state.product_data = summary_text
            st.success("Found matching products!")

            # Display summary
            st.subheader("📝 Customer Request Summary")
            st.write(summary_text)

            # Load products and get recommendations using ORIGINAL query for better accuracy
            st.subheader("🎯 Recommended Products")
            df = load_product_data()
            
            recommended_df = get_recommended_products(description, df, top_n=3)  # Use original description, not summary

            if recommended_df.empty:
                st.info("No recommended products found.")
            else:
                for idx, row in recommended_df.iterrows():
                    name = row.get("productDisplayName") or "Unnamed Product"
                    article_type = row.get("articleType") or ""
                    colour = row.get("baseColour") or ""
                    usage = row.get("usage") or ""
                    link = row.get("link") or ""
                    filename = row.get("filename")  # image filename

                    st.markdown(f"**{name}**")
                    details = " | ".join(filter(None, [article_type, colour, usage]))
                    if details:
                        st.write(details)
                    
                    # Load image with clean display
                    if link and isinstance(link, str) and link.strip():
                        try:
                            # Use use_column_width=True for better responsive design
                            st.image(link, use_column_width=True)
                        except Exception as e:
                            # If image fails to load, show a placeholder
                            st.image("https://via.placeholder.com/300x400?text=Product+Image", use_column_width=True)
                            st.caption(f"Image unavailable: {filename}")
                    else:
                        # No image link available
                        st.image("https://via.placeholder.com/300x400?text=No+Image", use_column_width=True)
                        st.caption("No image available")
                    
                    st.markdown("---")



def customer_browse_page():
    """Customer Browse by Category page (shop-style grid)"""
    st.title("🛍 Browse by Category")

    

    df = load_product_data()
    if df is None or df.empty:
        st.info("No products available.")
        return

    # --- Filter bar ---
    with st.container():
        st.markdown("### 🔎 Filter Products")
        col1, col2, col3, col4 = st.columns(4)

        # Category filter
        categories = sorted(df['masterCategory'].dropna().unique().tolist())
        selected_category = col1.selectbox("Category", ["All"] + categories)

    
        # Color filter
        available_colors = sorted(df['baseColour'].dropna().unique().tolist())
        selected_color = col4.selectbox("Color", ["All"] + available_colors)

    # --- Apply filters ---
    if selected_category != "All":
        df = df[df['masterCategory'] == selected_category]
    
        df = df[df['baseColour'] == selected_color]

    st.markdown("---")

    # --- Show products in grid (3 per row) ---
    if df.empty:
        st.info("No products found.")
        return

    num_cols = 3
    for i in range(0, len(df), num_cols):
        cols = st.columns(num_cols)
        for j, product in enumerate(df.iloc[i:i+num_cols].iterrows()):
            _, prod = product
            with cols[j]:
                # Product image
                if 'link' in prod and isinstance(prod['link'], str) and prod['link'].strip():
                    st.image(prod['link'], width=200)
                else:
                    st.image("https://via.placeholder.com/300x400?text=No+Image", use_container_width=True)

                # Product title
                st.markdown(f"**{prod.get('productDisplayName', 'Unnamed Product')}**")

                # Price
                price = prod.get("price", None)
                if price:
                    st.write(f"💰 LKR {price:,.2f}")
                else:
                    st.write("💰 Price on request")
                
                 # Tags (inside grid card)
                if 'tags' in prod and isinstance(prod['tags'], list) and len(prod['tags']) > 0:
                    tag_buttons = []
                    for i, t in enumerate(prod['tags']):
                        key = f"tag_{prod['id']}_{i}"
                        if st.button(f"#{t}", key=key):
                            st.session_state.selected_tag = t
                            st.session_state.customer_page = 'tag_products'
                            st.rerun()

                # View details button
                if st.button("View Details", key=f"view_{prod['id']}"):
                    st.session_state.selected_product = prod.to_dict()
                    st.session_state.customer_page = 'product_detail'
                    st.rerun()


def customer_tags_page():
    """Browse popular tags"""
    st.title("🏷️ Popular Tags")
    
    if st.button("← Back to Home"):
        st.session_state.customer_page = 'home'
        st.rerun()
    
    df = load_product_data()
    
    # Extract popular tags
    all_tags = []
    for tags in df['tags']:
        if isinstance(tags, list):
            all_tags.extend(tags)
    
    tag_counts = pd.Series(all_tags).value_counts()
    
    st.subheader("Most Popular Tags")
    cols = st.columns(4)
    for i, (tag, count) in enumerate(tag_counts.head(20).items()):
        col_idx = i % 4
        if cols[col_idx].button(f"{tag} ({count})", key=f"popular_{tag}"):
            st.session_state.selected_tag = tag
            st.session_state.customer_page = 'tag_products'
            st.rerun()

def customer_category_products_page():
    """Show products for a specific category"""
    st.title(f"📂 Products in: {st.session_state.selected_category}")
    
    if st.button("← Back to Browse"):
        st.session_state.customer_page = 'browse'
        st.session_state.selected_category = None
        st.session_state.selected_subcategory = None
        st.session_state.selected_article_type = None
        st.rerun()
    
    df = load_product_data()
    
    # Subcategory filter
    subcategories = get_subcategories(st.session_state.selected_category)
    selected_subcategory = st.selectbox("Filter by subcategory", ["All"] + subcategories)
    
    if selected_subcategory and selected_subcategory != "All":
        st.session_state.selected_subcategory = selected_subcategory
        
        # Article type filter
        article_types = get_article_types(selected_subcategory)
        selected_article_type = st.selectbox("Filter by type", ["All"] + article_types)
        
        if selected_article_type and selected_article_type != "All":
            st.session_state.selected_article_type = selected_article_type
            display_products(df, category=st.session_state.selected_category, 
                           subcategory=st.session_state.selected_subcategory,
                           article_type=st.session_state.selected_article_type)
        else:
            display_products(df, category=st.session_state.selected_category, 
                           subcategory=st.session_state.selected_subcategory)
    else:
        display_products(df, category=st.session_state.selected_category)

def customer_tag_products_page():
    """Show products with a specific tag"""
    st.title(f"🏷️ Products with tag: #{st.session_state.selected_tag}")
    
    if st.button("← Back"):
        st.session_state.customer_page = 'tags'
        st.session_state.selected_tag = None
        st.rerun()
    
    df = load_product_data()
    display_products(df, tag=st.session_state.selected_tag)

def customer_product_detail_page():
    """Show detailed product information"""
    if 'selected_product' not in st.session_state:
        st.error("No product selected")
        st.session_state.customer_page = 'home'
        st.rerun()
        return
    
    product = st.session_state.selected_product
    st.title(f"📦 {product.get('productDisplayName', 'Product Details')}")
    
    if st.button("← Back to Results"):
        st.session_state.customer_page = 'home'
        st.rerun()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Product Details")
        st.write(f"**Name:** {product.get('productDisplayName', 'No name')}")
        
        if 'masterCategory' in product:
            st.write(f"**Category:** {product.get('masterCategory', 'Unknown')}")
        if 'subCategory' in product:
            st.write(f"**Subcategory:** {product.get('subCategory', 'Unknown')}")
        if 'articleType' in product:
            st.write(f"**Type:** {product.get('articleType', 'Unknown')}")
        if 'baseColour' in product:
            st.write(f"**Color:** {product.get('baseColour', 'Unknown')}")
        if 'gender' in product:
            st.write(f"**Gender:** {product.get('gender', 'Unknown')}")
        if 'season' in product:
            st.write(f"**Season:** {product.get('season', 'Unknown')}")
        if 'usage' in product:
            st.write(f"**Usage:** {product.get('usage', 'Unknown')}")
        if 'year' in product and pd.notna(product['year']):
            st.write(f"**Year:** {int(product['year'])}")
        
        # Display tags if available
        if 'tags' in product and pd.notna(product['tags']):
            st.write("**Tags:**")
            tags = product['tags']
            
            if isinstance(tags, list):
                for tag in tags:
                    if st.button(f"#{tag}", key=f"detail_tag_{tag}"):
                        st.session_state.selected_tag = tag
                        st.session_state.customer_page = 'tag_products'
                        st.rerun()
    
    with col2:
        # Display product image if available
        if 'link' in product and pd.notna(product['link']):
            st.image(product['link'], use_container_width=True, caption="Product Image")
        elif 'filename' in product and pd.notna(product['filename']):
            st.image(f"https://via.placeholder.com/300x400?text=Product+Image", 
                    use_container_width=True, caption="Image not available")
        
        st.subheader("Actions")
        if st.button("Add to Cart", type="primary"):
            st.success("Added to cart!")
        
        if st.button("Save for Later"):
            st.info("Product saved!")

# ==================== ADMIN PAGES ====================

def admin_dashboard_page():
    """Admin dashboard overview"""
    st.title("👨‍💼 Admin Dashboard")
    st.write("Welcome to the admin panel. Manage your product catalog efficiently.")
    
    # Load data
    if st.session_state.admin_data is None:
        st.session_state.admin_data = load_product_data()
    
    df = st.session_state.admin_data
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(df))
    with col2:
        st.metric("Categories", df['masterCategory'].nunique())
    with col3:
        st.metric("Subcategories", df['subCategory'].nunique())
    with col4:
        st.metric("Genders", df['gender'].nunique())
    
    # Recent activity section
    st.subheader("📊 Quick Actions")
    
    action_cols = st.columns(3)
    
    with action_cols[0]:
        if st.button("📋 View All Products", use_container_width=True):
            st.session_state.admin_page = 'products'
            st.rerun()
    
    with action_cols[1]:
        if st.button("📈 Analytics", use_container_width=True):
            st.session_state.admin_page = 'analytics'
            st.rerun()
    
    with action_cols[2]:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.admin_page = 'settings'
            st.rerun()
    
    # Recent products
    st.subheader("🆕 Recent Products")
    if not df.empty:
        recent_products = df.head(5)
        for _, product in recent_products.iterrows():
            st.write(f"**{product.get('productDisplayName', 'Unnamed')}** - {product.get('masterCategory', 'Unknown')}")
    else:
        st.info("No products available")

def admin_products_page():
    """Admin product management"""
    st.title("📋 Product Management")
    
    if st.button("← Back to Dashboard"):
        st.session_state.admin_page = 'dashboard'
        st.rerun()
    
    # Load data
    if st.session_state.admin_data is None:
        st.session_state.admin_data = load_product_data()
    
    # Search functionality
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("Search products:", placeholder="Enter product name or description")
    with col2:
        categories = ["All"] + get_categories()
        selected_category = st.selectbox("Filter by category", categories)
    with col3:
        genders = ["All", "Men", "Women", "Unisex", "Boys", "Girls"]
        selected_gender = st.selectbox("Filter by gender", genders)
    
    if st.button("Search", type="primary"):
        with st.spinner("Searching products..."):
            results = st.session_state.admin_data.copy()
            
            if search_query:
                mask = results['productDisplayName'].str.contains(search_query, case=False, na=False)
                results = results[mask]
            
            if selected_category and selected_category != "All":
                results = results[results['masterCategory'] == selected_category]
            
            if selected_gender and selected_gender != "All":
                results = results[results['gender'] == selected_gender]
            
            st.session_state.search_results = results
    
    # Display search results
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        st.subheader(f"📊 Search Results ({len(results)} products)")
        
        if not results.empty:
            # Data table
            st.dataframe(results[['id', 'productDisplayName', 'masterCategory', 'subCategory', 'articleType', 'gender', 'baseColour']], 
                        use_container_width=True)
            
            # Export option
            if st.button("📤 Export Results"):
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="product_search_results.csv",
                    mime="text/csv"
                )
        else:
            st.info("No products found matching your criteria.")
    
    # Data management
    st.markdown("---")
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Data", help="Reload product data from source"):
            st.session_state.admin_data = load_product_data()
            st.session_state.search_results = None
            st.success("Data refreshed!")
    
    with col2:
        if st.button("📊 Show Full Catalog"):
            st.session_state.search_results = st.session_state.admin_data
            st.rerun()

def admin_analytics_page():
    """Admin analytics page"""
    st.title("📈 Analytics & Insights")
    
    if st.button("← Back to Dashboard"):
        st.session_state.admin_page = 'dashboard'
        st.rerun()
    
    df = st.session_state.admin_data
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        if 'masterCategory' in df:
            category_counts = df['masterCategory'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index, 
                         title="Product Distribution by Category")
            st.plotly_chart(fig)
    
    with col2:
        # Gender distribution
        if 'gender' in df:
            gender_counts = df['gender'].value_counts()
            fig = px.pie(values=gender_counts.values, names=gender_counts.index, 
                         title="Product Distribution by Gender")
            st.plotly_chart(fig)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Season distribution
        if 'season' in df:
            season_counts = df['season'].value_counts()
            fig = px.bar(x=season_counts.index, y=season_counts.values, 
                        title="Products by Season", labels={'x': 'Season', 'y': 'Count'})
            st.plotly_chart(fig)
    
    with col4:
        # Usage distribution
        if 'usage' in df:
            usage_counts = df['usage'].value_counts()
            fig = px.bar(x=usage_counts.index, y=usage_counts.values, 
                        title="Products by Usage", labels={'x': 'Usage', 'y': 'Count'})
            st.plotly_chart(fig)

def admin_settings_page():
    """Admin settings page"""
    st.title("⚙️ System Settings")
    
    if st.button("← Back to Dashboard"):
        st.session_state.admin_page = 'dashboard'
        st.rerun()
    
    st.subheader("Data Source Configuration")
    
    current_path = st.text_input("Data File Path", value=SELECTED_DATA_PATH or "(auto)")
    
    st.subheader("API Configuration")
    api_url = st.text_input("API Base URL", value=API_BASE_URL)
    
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")
        # Note: In a real application, you'd save these to a config file
    
    st.subheader("System Information")
    st.write(f"**Python Version:** {os.sys.version}")
    st.write(f"**Pandas Version:** {pd.__version__}")
    st.write(f"**Streamlit Version:** {st.__version__}")

# ==================== MAIN APPLICATION ====================

def render_customer_view():
    """Render the appropriate customer page"""
    if st.session_state.customer_page == 'home':
        customer_home_page()
    elif st.session_state.customer_page == 'ai_search':
        customer_ai_search_page()
    elif st.session_state.customer_page == 'browse':
        customer_browse_page()
    elif st.session_state.customer_page == 'tags':
        customer_tags_page()
    elif st.session_state.customer_page == 'category_products':
        customer_category_products_page()
    elif st.session_state.customer_page == 'tag_products':
        customer_tag_products_page()
    elif st.session_state.customer_page == 'product_detail':
        customer_product_detail_page()

def render_admin_view():
    """Render the appropriate admin page"""
    if st.session_state.admin_page == 'dashboard':
        admin_dashboard_page()
    elif st.session_state.admin_page == 'products':
        admin_products_page()
    elif st.session_state.admin_page == 'analytics':
        admin_analytics_page()
    elif st.session_state.admin_page == 'settings':
        admin_settings_page()

def main():
    """Main application"""
    # Sidebar navigation
    with st.sidebar:
        st.title("🛍️ Product AI")
        st.markdown("---")
        
        # Main navigation
        view_type = st.radio("Select View", ["Customer", "Admin"])
        
        if view_type == "Customer":
            st.session_state.current_page = 'customer_home'
            # Customer sub-navigation
            st.markdown("### Customer Pages")
            customer_pages = ["Home", "AI Search", "Browse", "Tags"]
            selected_customer_page = st.radio("Go to", customer_pages, key="customer_nav")
            
            if selected_customer_page == "Home":
                st.session_state.customer_page = 'home'
            elif selected_customer_page == "AI Search":
                st.session_state.customer_page = 'ai_search'
            elif selected_customer_page == "Browse":
                st.session_state.customer_page = 'browse'
            elif selected_customer_page == "Tags":
                st.session_state.customer_page = 'tags'
                
        else:
            st.session_state.current_page = 'admin_dashboard'
            # Admin sub-navigation
            st.markdown("### Admin Pages")
            admin_pages = ["Dashboard", "Products", "Analytics", "Settings"]
            selected_admin_page = st.radio("Go to", admin_pages, key="admin_nav")
            
            if selected_admin_page == "Dashboard":
                st.session_state.admin_page = 'dashboard'
            elif selected_admin_page == "Products":
                st.session_state.admin_page = 'products'
            elif selected_admin_page == "Analytics":
                st.session_state.admin_page = 'analytics'
            elif selected_admin_page == "Settings":
                st.session_state.admin_page = 'settings'
        
        st.markdown("---")
        st.write("**API Status:**")
        try:
            health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
            st.success("✅ API Connected")
            st.caption(f"Model: {health.get('model', 'Unknown')}")
        except:
            st.error("❌ API Offline")
            st.info("Running in demo mode with sample data")
        
        # Data status
        st.write("**Data Status:**")
        try:
            df = load_product_data()
            st.success(f"✅ Loaded {len(df)} products")
            st.caption(f"Categories: {df['masterCategory'].nunique()}")
        except:
            st.error("❌ Data load failed")
        
        st.markdown("---")
        st.caption(f"© {datetime.now().year} Product AI System")

    # Render the appropriate view
    if st.session_state.current_page == 'customer_home':
        render_customer_view()
    else:
        render_admin_view()

if __name__ == "__main__":
    main()