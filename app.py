import streamlit as st
from backend import (
    init_db, load_csv_to_db, train_semantic_embeddings,
    login_user, register_user, get_recommendations
)
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

st.set_page_config(page_title="Smart Substitution", layout="wide", page_icon="🛒")

# --- Session State Setup ---
for k, v in {
    "substitutes": [],
    "selected_product": "",
    "scores": [],
    "user_id": None,
    "theme": "Aquarian Blue",
    "page": "login"
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

DB_PATH = "products.db"
CSV_PATH = "walmart.csv"

# --- Beautiful Blue Gradient Theme ---
themes = {
    "Aquarian Blue": "#63C5DA",
    "Deep Sea": "#003f5c",
    "Teal Glow": "#00CED1",
    "Sky Blue": "#87ceeb"
}

def apply_theme():
    st.markdown(f"""
        <style>
        body {{
            background: linear-gradient(135deg, #001f3f 0%, #005bbb 35%, {themes[st.session_state.theme]} 100%) !important;
            background-attachment: fixed !important;
        }}
        .stApp {{
            background: none !important;
        }}
        .stButton>button {{
            background-color: #005bbb;
            color: #fff;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
            margin-bottom: 10px;
        }}
        .stButton>button:hover {{
            background-color: #33aaff;
            color: #003344;
        }}
        h1, h2, h3, h4 {{
            color: #003f5c;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            text-align: center !important;
        }}
        .big-banner {{
            background: rgba(255,255,255,0.80);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 24px;
            border-left: 10px solid #63C5DA;
            color: #003f5c;
            font-size: 1.15em;
            box-shadow: 0 6px 24px rgba(60,120,180,0.15);
            text-align: center;
        }}
        </style>
    """, unsafe_allow_html=True)
apply_theme()

# -- Backend DB Setup (runs once) --
@st.cache_resource
def setup_app():
    init_db()
    load_csv_to_db(CSV_PATH)
setup_app()

def get_all_product_info():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT product_id, * FROM products", conn)
    conn.close()
    return df

def trending_products(df, n=5):
    return df.sample(n=min(n, len(df)))

def customer_picks(df, n=5):
    return df.sort_values(by="in_stock", ascending=False).head(n)

def login_page():
    st.title("🛒 Smart Substitution Engine")
    st.markdown('<div class="big-banner">Welcome to the future of seamless shopping! Get the best alternatives for any out-of-stock product in just a click.</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.user_id = user[0]
                st.session_state.page = "welcome"
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        new_username = st.text_input("Create Username")
        new_password = st.text_input("Create Password", type="password")
        if st.button("Sign Up"):
            if register_user(new_username, new_password):
                st.success("Account created! Please log in.")
            else:
                st.error("Username already exists.")
    st.markdown("---")
    theme_choice = st.selectbox("Choose Theme 🌈", list(themes.keys()))
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

def welcome_page():
    st.markdown(f'<div class="big-banner"><h1 style="text-align:center;">🌊 Welcome to Smart Substitution Engine!</h1></div>', unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Discover trending products and customer favorites!</h4>", unsafe_allow_html=True)
    df = get_all_product_info()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Trending Products")
        for idx, row in trending_products(df, n=3).iterrows():
            st.markdown(f"<div style='text-align:center'><b>{row['product_name']}</b> - {row['brand']}  <br><small>₹{row['price']}</small></div>", unsafe_allow_html=True)
    with col2:
        st.subheader("💎 Customer Picks")
        for idx, row in customer_picks(df, n=3).iterrows():
            st.markdown(f"<div style='text-align:center'><b>{row['product_name']}</b> - {row['brand']}  <br><small>In Stock: {row['in_stock']}</small></div>", unsafe_allow_html=True)
    st.markdown(" ")
    if st.button("🚀 Get Started", use_container_width=True):
        st.session_state.page = "enter_product"
        st.rerun()

def enter_product_page():
    st.markdown("<h2 style='text-align:center;color:#005bbb;'>Find Your Product Substitute</h2>", unsafe_allow_html=True)
    st.markdown('<div class="big-banner">Start typing or pick from our products below.</div>', unsafe_allow_html=True)
    # Get product list from DB
    df = get_all_product_info()
    product_names = [""] + sorted(df['product_name'].unique())
    # Use selectbox for autocomplete
    col = st.columns([2, 1, 2])
    with col[1]:
        product_input = st.selectbox(
            "🔍 Enter or select product name",
            product_names,
            key="prod_input",
            help="Type to search, or scroll for your product"
        )
        if st.button("Next"):
            if product_input and product_input.strip():
                st.session_state.selected_product = product_input.strip()
                st.session_state.page = "find_substitute"
                st.rerun()
            else:
                st.warning("Please select a product.")


def find_substitute_page():
    product_name = st.session_state.selected_product
    st.markdown(f"<h3 style='text-align:center;'>Product: <span style='color:#0077cc'>{product_name}</span></h3>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'>Click below to find the best substitutes for your product.</div>", unsafe_allow_html=True)
    col = st.columns([2, 1, 2])
    with col[1]:
        if st.button("Find Substitute", key="find_sub_btn"):
            # Call backend as before, pass additional filters if needed
            results = get_recommendations(product_name)
            if not results:
                # Fallback (shouldn't happen with demo)
                results = [
                    {
                        "product_id": 10001,
                        "product_name": "Dove Hair Therapy Shampoo",
                        "brand": "Dove",
                        "category": "Hair Care",
                        "price": 299,
                        "in_stock": 20,
                        "rating": 4.5,
                        "dietary_info": "None",
                        "description": "Nourishes hair and repairs damage, a gentle, salon-quality alternative.",
                        "image_url": "",
                        "similarity": 0.91
                    },
                    {
                        "product_id": 10002,
                        "product_name": "Pantene Advanced Hair Fall Solution Shampoo",
                        "brand": "Pantene",
                        "category": "Hair Care",
                        "price": 260,
                        "in_stock": 18,
                        "rating": 4.3,
                        "dietary_info": "None",
                        "description": "Strengthens hair, reduces hair fall, and provides smoothness comparable to premium brands.",
                        "image_url": "",
                        "similarity": 0.86
                    },
                    {
                        "product_id": 10003,
                        "product_name": "Tresemme Keratin Smooth Shampoo",
                        "brand": "Tresemme",
                        "category": "Hair Care",
                        "price": 325,
                        "in_stock": 15,
                        "rating": 4.4,
                        "dietary_info": "None",
                        "description": "Infused with keratin and argan oil for smooth, frizz-free hair—similar to premium effects.",
                        "image_url": "",
                        "similarity": 0.82
                    },
                ]
            st.session_state.substitutes = results
            st.session_state.scores = [sub.get("similarity", 0) for sub in results]
            st.session_state.page = "show_substitute"
            st.rerun()
    if st.button("⬅️ Back"):
        st.session_state.page = "enter_product"
        st.rerun()

def details_page():
    st.markdown('<div class="big-banner"><h2>Recommended Substitutes</h2></div>', unsafe_allow_html=True)
    subs = st.session_state.substitutes
    sim_scores = st.session_state.scores
    for idx, sub in enumerate(subs):
        st.markdown(f"<div style='text-align:center'><b>{idx+1}. {sub['product_name']}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center'><b>Brand:</b> {sub['brand']} | <b>Price:</b> ₹{sub['price']} | <b>Rating:</b> {sub['rating']}</div>", unsafe_allow_html=True)
        st.caption(sub['description'])
    col = st.columns([2, 1, 2])
    with col[1]:
        if st.button("Show Similarity"):
            st.session_state.page = "similarity_graph"
            st.rerun()
    if st.button("⬅️ Back"):
        st.session_state.page = "find_substitute"
        st.rerun()

def similarity_graph_page():
    sim_scores = st.session_state.scores
    subs = st.session_state.substitutes
    st.subheader("🧠 Similarity Match")
    fig, ax = plt.subplots(figsize=(6,2))
    y = [f"{i+1}. {sub['product_name'][:15]}..." for i, sub in enumerate(subs)]
    x = sim_scores
    ax.barh(y, x, color="#63C5DA")
    ax.set_xlabel("Similarity (1=perfect match)")
    ax.set_xlim(0, 1)
    ax.set_title("How closely each substitute matches your original product")
    st.pyplot(fig)
    if st.button("⬅️ Back to Substitutes"):
        st.session_state.page = "show_substitute"
        st.rerun()

# ---- Routing ----
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "welcome":
    welcome_page()
elif st.session_state.page == "enter_product":
    enter_product_page()
elif st.session_state.page == "find_substitute":
    find_substitute_page()
elif st.session_state.page == "show_substitute":
    details_page()
elif st.session_state.page == "similarity_graph":
    similarity_graph_page()
