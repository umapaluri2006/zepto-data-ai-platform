# -*- coding: utf-8 -*-
"""
data_pipeline/pipeline.py
End-to-end Data Engineering pipeline:
1. Scrape books from books.toscrape.com
2. Clean, type-cast, and enrich fields (fixed rate: 1 GBP = 105.50 INR)
3. Load into a normalized 2-table SQLite database with PK/FK
4. Run 5 distinct SQL queries covering required clauses & JOIN
5. Validate via pandas pd.read_sql and in-memory pd.merge
"""

import sys
import io
import os
import re
import sqlite3
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from scraper import scrape_books

# Project baseline fixed conversion rate (1 GBP = 105.50 INR)
EXCHANGE_RATE_GBP_TO_INR = 105.50
DB_PATH = os.path.join(os.path.dirname(__file__), "books.db")

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5
}

def clean_and_transform(raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Cleans raw scraped book data into typed, enriched DataFrame.
    """
    df = pd.DataFrame(raw_records)
    
    # 1. Clean Price GBP
    def parse_price(val):
        if pd.isna(val):
            return np.nan
        match = re.search(r"(\d+\.?\d*)", str(val))
        return float(match.group(1)) if match else np.nan

    df["price_gbp"] = df["price"].apply(parse_price)
    
    # Impute missing prices with median if any unparsable (justified to prevent pipeline failure)
    if df["price_gbp"].isna().any():
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)
        
    df["price_gbp"] = df["price_gbp"].astype(float).round(2)
    
    # 2. Convert to price_inr using fixed rate (1 GBP = 105.50 INR)
    df["price_inr"] = (df["price_gbp"] * EXCHANGE_RATE_GBP_TO_INR).round(2)
    
    # 3. Clean rating to integer (1-5)
    def parse_rating(val):
        if pd.isna(val):
            return 3 # default median rating
        key = str(val).strip().lower()
        return RATING_MAP.get(key, 3)
        
    df["rating"] = df["star_rating"].apply(parse_rating).astype(int)
    
    # 4. Parse availability to boolean
    def parse_availability(val):
        if pd.isna(val):
            return True
        return "in stock" in str(val).lower()
        
    df["in_stock"] = df["availability"].apply(parse_availability).astype(bool)
    
    # Clean titles and categories
    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    
    cleaned_df = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]].copy()
    return cleaned_df

def init_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Creates a normalized 2-table SQLite schema with PK/FK constraints."""
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories (category_id) ON DELETE CASCADE
    );
    """)
    
    conn.commit()
    return conn

def load_data_to_db(df: pd.DataFrame, conn: sqlite3.Connection):
    """Loads cleaned data into categories and books normalized tables."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Insert unique categories
    unique_categories = sorted(df["category"].unique())
    for cat in unique_categories:
        cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,))
    conn.commit()
    
    # Map category names to category_id
    cat_df = pd.read_sql("SELECT category_id, category_name FROM categories", conn)
    cat_map = dict(zip(cat_df["category_name"], cat_df["category_id"]))
    
    # Prepare books records
    books_records = []
    for _, row in df.iterrows():
        books_records.append((
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            1 if row["in_stock"] else 0,
            cat_map[row["category"]]
        ))
        
    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?)
    """, books_records)
    conn.commit()
    print(f"Successfully loaded {len(unique_categories)} categories and {len(books_records)} books into SQLite database.")

def execute_sql_queries(conn: sqlite3.Connection) -> Dict[str, pd.DataFrame]:
    """
    Executes at least 5 SQL queries covering required SQL clauses:
    1. SELECT / WHERE
    2. ORDER BY
    3. LIMIT
    4. DISTINCT
    5. IN / BETWEEN
    + JOIN between categories and books.
    """
    queries = {
        "Query 1 (SELECT, WHERE, ORDER BY) - Highly rated books (Rating >= 4)": """
            SELECT title, price_gbp, price_inr, rating
            FROM books
            WHERE rating >= 4 AND in_stock = 1
            ORDER BY rating DESC, price_gbp ASC;
        """,
        
        "Query 2 (ORDER BY, LIMIT) - Top 5 most expensive books": """
            SELECT title, price_gbp, price_inr, rating
            FROM books
            ORDER BY price_inr DESC
            LIMIT 5;
        """,
        
        "Query 3 (DISTINCT) - Distinct ratings present in catalog": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating ASC;
        """,
        
        "Query 4 (BETWEEN & WHERE) - Books in budget price range (INR 2000 to 4000)": """
            SELECT title, price_inr, rating
            FROM books
            WHERE price_inr BETWEEN 2000.0 AND 4000.0
            ORDER BY price_inr ASC;
        """,
        
        "Query 5 (JOIN, GROUP BY, ORDER BY) - Category summary with average price & book count": """
            SELECT 
                c.category_name,
                COUNT(b.book_id) AS total_books,
                ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
                ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
                MAX(b.rating) AS max_rating
            FROM categories c
            INNER JOIN books b ON c.category_id = b.category_id
            GROUP BY c.category_id, c.category_name
            ORDER BY total_books DESC, avg_price_inr DESC;
        """
    }
    
    results = {}
    print("\n" + "="*80)
    print("EXECUTING REQUIRED SQL QUERIES")
    print("="*80)
    
    for label, query in queries.items():
        print(f"\n>>> {label}")
        print("SQL:")
        print(query.strip())
        df_res = pd.read_sql_query(query, conn)
        print("\nResult (first 5 rows):")
        print(df_res.head(5).to_string(index=False))
        print(f"Total rows returned: {len(df_res)}")
        results[label] = df_res
        
    return results

def verify_pandas_merge_equivalence(conn: sqlite3.Connection):
    """
    Reads back tables into pandas DataFrames and reproduces the SQL JOIN query
    using in-memory pd.merge to demonstrate equivalent output.
    """
    print("\n" + "="*80)
    print("VERIFYING SQL JOIN VS PANDAS pd.merge EQUIVALENCE")
    print("="*80)
    
    # 1. SQL Join output
    sql_join_query = """
        SELECT 
            c.category_name,
            COUNT(b.book_id) AS total_books,
            ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
            ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
            MAX(b.rating) AS max_rating
        FROM categories c
        INNER JOIN books b ON c.category_id = b.category_id
        GROUP BY c.category_id, c.category_name
        ORDER BY total_books DESC, avg_price_inr DESC;
    """
    df_sql = pd.read_sql_query(sql_join_query, conn)
    
    # 2. In-memory pandas merge
    df_books = pd.read_sql("SELECT * FROM books", conn)
    df_categories = pd.read_sql("SELECT * FROM categories", conn)
    
    merged = pd.merge(df_categories, df_books, on="category_id", how="inner")
    df_pandas = merged.groupby(["category_id", "category_name"]).agg(
        total_books=("book_id", "count"),
        avg_price_gbp=("price_gbp", lambda x: round(x.mean(), 2)),
        avg_price_inr=("price_inr", lambda x: round(x.mean(), 2)),
        max_rating=("rating", "max")
    ).reset_index().drop(columns=["category_id"])
    
    df_pandas = df_pandas.sort_values(by=["total_books", "avg_price_inr"], ascending=[False, False]).reset_index(drop=True)
    
    print("\n--- SQL JOIN RESULT ---")
    print(df_sql.to_string(index=False))
    
    print("\n--- PANDAS MERGE & AGGREGATION RESULT ---")
    print(df_pandas.to_string(index=False))
    
    # Verify equivalence
    pd.testing.assert_frame_equal(df_sql, df_pandas, check_dtype=False)
    print("\n[SUCCESS] Exact equivalence confirmed between SQL JOIN and pandas.merge()!")

def run_pipeline() -> Tuple[pd.DataFrame, sqlite3.Connection]:
    """Runs the complete Module 1 pipeline."""
    print("Starting Data Engineering Pipeline...")
    raw_books = scrape_books()
    cleaned_df = clean_and_transform(raw_books)
    
    print(f"\nCleaned Data Summary:")
    print(cleaned_df.info())
    
    conn = init_database(DB_PATH)
    load_data_to_db(cleaned_df, conn)
    execute_sql_queries(conn)
    verify_pandas_merge_equivalence(conn)
    
    return cleaned_df, conn

if __name__ == "__main__":
    df, conn = run_pipeline()
    conn.close()
    print("\nModule 1 Pipeline completed successfully!")
