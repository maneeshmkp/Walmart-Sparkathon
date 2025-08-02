import sqlite3
import pandas as pd
from backend import train_semantic_embeddings

DB_NAME = "products.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT,
            brand TEXT,
            category TEXT,
            price REAL,
            in_stock INTEGER,
            rating REAL,
            dietary_info TEXT,
            description TEXT,
            image_url TEXT
        )
    """)
    # ... (rest as before)
    conn.commit()
    conn.close()


def load_csv_to_db(csv_file):
    df = pd.read_csv(csv_file)
    df = df[["Product Name", "Brand", "Sale Price", "Category", "Available"]].dropna(subset=["Product Name"])
    df = df.rename(columns={
        "Product Name": "product_name",
        "Brand": "brand",
        "Sale Price": "price",
        "Category": "category",
        "Available": "in_stock"
    })
    df["rating"] = 4.0
    df["dietary"] = "None"
    df["image_url"] = ""

    conn = sqlite3.connect("products.db")
    df.to_sql("products", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    load_csv_to_db("walmart.csv")  # Make sure this file is in your project directory
    train_semantic_embeddings()
    print("✅ Database initialized and trained!")
