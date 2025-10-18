"""
Streamlit App for Product Categorization AI
Optimized for Streamlit Cloud deployment
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import ast
import sys
import re
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Try to import ollama, but don't fail if not available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    st.warning("⚠️ Ollama not available. Some AI features may be limited.")

# Configuration
API_BASE_URL = "http://localhost:8000/api"  # Will be updated for production

# Resolve data path dynamically
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "backend", "data", "cleaned_product_data.csv"),
    os.path.join(PROJECT_ROOT, "backend", "data", "product.csv"),
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
        # Pick the first existing path from candidates
        data_path = None
        for candidate in DATA_PATH_CANDIDATES:
            if os.path.exists(candidate):
                data_path = candidate
                break

        if data_path and os.path.exists(data_path):
            # Store selected path for display
            global SELECTED_DATA_PATH
            SELECTED_DATA_PATH = data_path
            df = pd.read_csv(data_path)
            
            # Create tags from available attributes for better discovery
            if 'gender' in df.columns:
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
    """Generic API call function with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=10)
        else:
            response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Backend API not available. Running in demo mode.")
        return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def generate_summary(description: str) -> str:
    """Generate summary using available AI tools"""
    if OLLAMA_AVAILABLE:
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
    else:
        return description  # Return original description if Ollama not available

def get_recommended_products(description, df, top_n=5):
    """Get recommended products based on description"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Simple keyword matching for demo
    description_lower = description.lower()
    matches = []
    
    for _, row in df.iterrows():
        score = 0
        product_text = f"{row.get('productDisplayName', '')} {row.get('articleType', '')} {row.get('baseColour', '')} {row.get('usage', '')}".lower()
        
        # Check for color matches
        colors = ['red', 'blue', 'green', 'black', 'white', 'brown', 'gray', 'grey', 'pink', 'purple', 'yellow', 'orange', 'navy']
        for color in colors:
            if color in description_lower and color in product_text:
                score += 2
        
        # Check for product type matches
        types = ['shirt', 'pants', 'jeans', 'shoes', 'dress', 'watch', 'bag']
        for ptype in types:
            if ptype in description_lower and ptype in product_text:
                score += 2
        
        if score > 0:
            row_dict = row.to_dict()
            row_dict['score'] = score
            row_dict['match_type'] = 'exact' if score >= 4 else 'related'
            matches.append(row_dict)
    
    # Sort by score and return top matches
    matches.sort(key=lambda x: x['score'], reverse=True)
    return pd.DataFrame(matches[:top_n])

# Import the main application functions from the original app
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
                    if prod.get('link') and prod['link'] != 'undefined' and prod['link'].strip():
                        try:
                            st.image(prod['link'], width=200)
                        except:
                            st.image("https://via.placeholder.com/200x200?text=No+Image", width=200)
                    else:
                        st.image("https://via.placeholder.com/200x200?text=No+Image", width=200)

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

        # Generate summary
        summary_text = generate_summary(description)
        st.session_state.product_data = summary_text
        st.success("Found matching products!")

        # Display summary
        st.subheader("📝 Customer Request Summary")
        st.write(summary_text)

        # Load products and get recommendations
        df = load_product_data()
        recommended_df = get_recommended_products(description, df, top_n=6)

        if recommended_df.empty:
            st.info("No recommended products found.")
        else:
            st.subheader("🎯 Recommended Products")
            display_products(recommended_df)

def admin_dashboard_page():
    """Admin dashboard overview"""
    st.title("👨‍💼 Admin Dashboard")
    st.write("Welcome to the admin panel. Manage your product catalog efficiently.")
    
    # Load data
    if st.session_state.admin_data is None:
        st.session_state.admin_data = load_product_data()
    
    df = st.session_state.admin_data
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Products", len(df))
    with col2:
        st.metric("Categories", df['masterCategory'].nunique())
    with col3:
        st.metric("Subcategories", df['subCategory'].nunique())
    
    # Recent activity section
    st.subheader("📊 Quick Actions")
    
    action_cols = st.columns(4)
    
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
    
    with action_cols[3]:
        if st.button("➕ Add Product", use_container_width=True):
            st.session_state.admin_page = 'add_product'
            st.rerun()
    
    # Recent products
    st.subheader("🆕 Recent Products")
    if not df.empty:
        recent_products = df.head(5)
        for _, product in recent_products.iterrows():
            st.write(f"**{product.get('productDisplayName', 'Unnamed')}** - {product.get('masterCategory', 'Unknown')}")
    else:
        st.info("No products available")

def admin_add_product_page():
    """Admin add new product page with MongoDB integration"""
    st.title("➕ Add New Product")
    
    if st.button("← Back to Dashboard"):
        st.session_state.admin_page = 'dashboard'
        st.rerun()
    
    st.subheader("Product Information")
    
    with st.form("add_product_form"):
        # Product Name (Required)
        product_name = st.text_input("Product Name*", placeholder="e.g., Nike Men Black T-shirt", help="Enter the name of the product")
        
        # Product Description (Required)
        product_description = st.text_area("Product Description*", placeholder="Describe the product (color, gender, category, subcategory, brand, usage, type).\nExample: 'Men black cotton T-shirt by Nike for casual wear, apparel/topwear.'", help="Describe the product features and details", height=100)
        
        # Image URL (Optional)
        image_url = st.text_input("Image URL", placeholder="https://.../image.jpg(optional)", help="Optional: Enter the URL of the product image")
        
        submitted = st.form_submit_button("Add Product with AI Categorization", type="primary")
    
    if submitted:
        if not product_name or not product_description:
            st.error("Product name and description are required!")
        else:
            # Simple categorization for demo
            with st.spinner("🤖 AI is analyzing the product description..."):
                # Simple rule-based categorization
                desc_lower = product_description.lower()
                
                # Extract gender - improved logic
                gender = 'Unisex'  # Default to Unisex instead of Men
                
                # Check for men indicators
                men_words = ['men', 'male', 'man', 'mens', 'guy', 'boys', 'boy', 'men\'s', 'mens\'', 'guys\'', 'boys\'']
                women_words = ['women', 'female', 'woman', 'womens', 'lady', 'ladies', 'girls', 'girl', 'women\'s', 'womens\'', 'ladies\'', 'girls\'', 'females']
                
                # Use word boundaries to avoid substring matches
                men_count = sum(1 for word in men_words if re.search(r'\b' + re.escape(word) + r'\b', desc_lower))
                women_count = sum(1 for word in women_words if re.search(r'\b' + re.escape(word) + r'\b', desc_lower))
                
                if men_count > women_count:
                    gender = 'Men'
                elif women_count > men_count:
                    gender = 'Women'
                elif men_count > 0 and women_count > 0:
                    gender = 'Unisex'  # Both mentioned
                else:
                    # Check product name as well
                    name_lower = product_name.lower()
                    name_men_count = sum(1 for word in men_words if re.search(r'\b' + re.escape(word) + r'\b', name_lower))
                    name_women_count = sum(1 for word in women_words if re.search(r'\b' + re.escape(word) + r'\b', name_lower))
                    
                    if name_men_count > name_women_count:
                        gender = 'Men'
                    elif name_women_count > name_men_count:
                        gender = 'Women'
                    else:
                        gender = 'Unisex'  # Default to Unisex when unclear
                
                # Extract color - improved logic
                colors = ['black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 'purple', 'brown', 'gray', 'grey', 'orange', 'navy', 'beige', 'tan', 'silver', 'gold', 'maroon', 'teal', 'coral', 'lime', 'indigo', 'violet', 'crimson', 'turquoise']
                color = 'Unknown'
                
                # Check both description and product name
                text_to_check = f"{desc_lower} {product_name.lower()}"
                
                for c in colors:
                    if c in text_to_check:
                        color = c.title()
                        break
                
                # Extract category - improved logic
                text_to_check = f"{desc_lower} {product_name.lower()}"
                
                # Footwear detection
                footwear_words = ['shoe', 'shoes', 'sneaker', 'sneakers', 'boot', 'boots', 'sandal', 'sandals', 'heel', 'heels', 'loafer', 'loafers', 'slipper', 'slippers']
                if any(word in text_to_check for word in footwear_words):
                    category = 'Footwear'
                    subcategory = 'Shoes'
                
                # Accessories detection
                elif any(word in text_to_check for word in ['watch', 'watches', 'bag', 'bags', 'handbag', 'purse', 'hat', 'cap', 'belt', 'belt', 'jewelry', 'necklace', 'ring', 'earring']):
                    category = 'Accessories'
                    if any(word in text_to_check for word in ['watch', 'watches']):
                        subcategory = 'Watches'
                    elif any(word in text_to_check for word in ['bag', 'bags', 'handbag', 'purse']):
                        subcategory = 'Bags'
                    else:
                        subcategory = 'Watches'  # Default for accessories
                
                # Apparel detection
                elif any(word in text_to_check for word in ['shirt', 't-shirt', 'tshirt', 'top', 'blouse', 'dress', 'pants', 'jeans', 'trouser', 'shorts', 'skirt', 'jacket', 'coat', 'sweater', 'hoodie']):
                    category = 'Apparel'
                    # Topwear detection
                    if any(word in text_to_check for word in ['shirt', 't-shirt', 'tshirt', 'top', 'blouse', 'dress', 'jacket', 'coat', 'sweater', 'hoodie']):
                        subcategory = 'Topwear'
                    # Bottomwear detection
                    elif any(word in text_to_check for word in ['pants', 'jeans', 'trouser', 'shorts', 'skirt']):
                        subcategory = 'Bottomwear'
                    else:
                        subcategory = 'Topwear'  # Default for apparel
                
                # Default fallback
                else:
                    category = 'Apparel'
                    subcategory = 'Topwear'
                
                # Create product data
                product_data = {
                    "productDisplayName": product_name,
                    "description": product_description,
                    "gender": gender,
                    "masterCategory": category,
                    "subCategory": subcategory,
                    "articleType": "General",
                    "baseColour": color,
                    "season": "All Season",
                    "year": 2024,
                    "usage": "Casual",
                    "link": image_url if image_url else '',
                    "filename": f"{product_name.replace(' ', '_').lower()}.jpg"
                }
                
                # Show AI reasoning for debugging
                st.info(f"🤖 AI Analysis: Detected '{gender}' gender, '{category}' category, '{color}' color")
                
                # Try to add to MongoDB via API
                result = call_api("/products/add", product_data)
                
                if result and result.get("success"):
                    st.success(f"✅ Product added successfully! ID: {result.get('product_id')}")
                    
                    # Show the added product details
                    st.subheader("🎉 Product Added Successfully!")
                    added_product = result.get("product", {})
                    
                    # Display in a nice format
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📋 Product Details:**")
                        st.write(f"**ID:** {added_product.get('id')}")
                        st.write(f"**Name:** {added_product.get('productDisplayName')}")
                        st.write(f"**Description:** {product_description}")
                        st.write(f"**Gender:** {added_product.get('gender') or 'Not specified'}")
                        st.write(f"**Category:** {added_product.get('masterCategory') or 'Not specified'}")
                    
                    with col2:
                        st.markdown("**🏷️ AI-Generated Attributes:**")
                        st.write(f"**Subcategory:** {added_product.get('subCategory') or 'Not specified'}")
                        st.write(f"**Type:** {added_product.get('articleType') or 'Not specified'}")
                        st.write(f"**Color:** {added_product.get('baseColour') or 'Not specified'}")
                        st.write(f"**Season:** {added_product.get('season') or 'Not specified'}")
                        st.write(f"**Usage:** {added_product.get('usage') or 'Not specified'}")
                        st.write(f"**Year:** {added_product.get('year') or 'Not specified'}")
                    
                    st.toast("🎉 Product added successfully!", icon="✅")
                    
                    if st.button("➕ Add Another Product"):
                        st.rerun()
                else:
                    st.warning("⚠️ Product categorization completed, but couldn't save to database (API not available)")
                    
                    # Show categorized product anyway
                    st.subheader("🎉 Product Categorized Successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📋 Product Details:**")
                        st.write(f"**Name:** {product_name}")
                        st.write(f"**Description:** {product_description}")
                        st.write(f"**Gender:** {gender}")
                        st.write(f"**Category:** {category}")
                    
                    with col2:
                        st.markdown("**🏷️ AI-Generated Attributes:**")
                        st.write(f"**Subcategory:** {subcategory}")
                        st.write(f"**Type:** General")
                        st.write(f"**Color:** {color}")
                        st.write(f"**Season:** All Season")
                        st.write(f"**Usage:** Casual")
                        st.write(f"**Year:** 2024")
                    
                    st.info("💡 In production, this would be saved to your MongoDB database.")

def render_customer_view():
    """Render the appropriate customer page"""
    if st.session_state.customer_page == 'home':
        customer_home_page()
    elif st.session_state.customer_page == 'ai_search':
        customer_ai_search_page()
    elif st.session_state.customer_page == 'browse':
        st.info("Browse functionality coming soon!")
    elif st.session_state.customer_page == 'tags':
        st.info("Tags functionality coming soon!")

def render_admin_view():
    """Render the appropriate admin page"""
    if st.session_state.admin_page == 'dashboard':
        admin_dashboard_page()
    elif st.session_state.admin_page == 'add_product':
        admin_add_product_page()
    elif st.session_state.admin_page == 'products':
        st.info("Product management coming soon!")
    elif st.session_state.admin_page == 'analytics':
        st.info("Analytics coming soon!")
    elif st.session_state.admin_page == 'settings':
        st.info("Settings coming soon!")

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
            admin_pages = ["Dashboard", "Add Product", "Products", "Analytics", "Settings"]
            selected_admin_page = st.radio("Go to", admin_pages, key="admin_nav")
            
            if selected_admin_page == "Dashboard":
                st.session_state.admin_page = 'dashboard'
            elif selected_admin_page == "Add Product":
                st.session_state.admin_page = 'add_product'
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
        except:
            st.warning("⚠️ API Offline")
            st.info("Running in demo mode")
        
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
